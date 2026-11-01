import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config, save_config
from LRAEP_trainer.config_export import write_open_text
from LRAEP_trainer.network import SocksProxy, routed
from LRAEP_trainer.telegram import TelegramUpdates


def configuration():
    value = copy.deepcopy(DEFAULT_CONFIG)
    value["trainer_name"] = "test trainer Ω"
    value["telegram"] = {
        "enabled": True, "tgBot_key": "123:PRIVATE_TEST_TOKEN", "tgChatID": -12345,
        "update_interval_seconds": 2,
    }
    value["vpn"]["link"] = "vless://fixture@example.invalid:443"
    return value


class TelegramConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.json"
        write_open_text(self.path.with_name(".env"), "offline test key")
        self.value = configuration()

    def test_settings_roundtrip_and_old_configs_default_to_disabled(self):
        save_config(self.path, self.value)
        loaded = load_config(self.path)
        self.assertEqual(loaded["trainer_name"], self.value["trainer_name"])
        self.assertEqual(loaded["telegram"], self.value["telegram"])
        self.value.pop("telegram")
        self.value.pop("trainer_name")
        save_config(self.path, self.value)
        loaded = load_config(self.path)
        self.assertFalse(loaded["telegram"]["enabled"])
        self.assertEqual(loaded["trainer_name"], DEFAULT_CONFIG["trainer_name"])

    def test_invalid_values_are_rejected_without_echoing_secrets(self):
        cases = [
            ("enabled", "PRIVATE_TEST_TOKEN"), ("tgBot_key", "PRIVATE_TEST_TOKEN"),
            ("tgBot_key", ""), ("tgChatID", ""), ("tgChatID", True),
            ("tgChatID", []), ("tgChatID", "PRIVATE_TEST_TOKEN"),
            ("update_interval_seconds", 0), ("update_interval_seconds", -1),
            ("update_interval_seconds", True), ("update_interval_seconds", "2"),
            ("update_interval_seconds", float("inf")),
            ("update_interval_seconds", float("nan")),
            ("unknown", "PRIVATE_TEST_TOKEN"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(self.value)
                config["telegram"][field] = value
                with self.assertRaises(ConfigError) as caught:
                    save_config(self.path, config)
                self.assertNotIn("PRIVATE_TEST_TOKEN", str(caught.exception))
        for value in ("", "  ", "bad\nname", None, 123, "x" * 129):
            with self.subTest(name=value), self.assertRaises(ConfigError):
                save_config(self.path, dict(self.value, trainer_name=value))


class TelegramUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.opener = MagicMock()
        self.opener.open.return_value.__enter__.return_value.read.return_value = b'{"ok":true}'

    def run_periods(self, updates, count=2):
        clock = [0.0]
        waits = []

        def wait(seconds):
            waits.append(seconds)
            if len(waits) > count:
                return True
            clock[0] += seconds
            return False

        with patch.object(updates, "_stopped") as stopped, \
                patch("LRAEP_trainer.telegram.time.monotonic", side_effect=lambda: clock[0]), \
                patch("LRAEP_trainer.telegram.http_opener", return_value=self.opener):
            stopped.wait.side_effect = wait
            updates._run()
        return waits

    def test_periodic_payload_has_name_latest_progress_and_configured_chat(self):
        snapshot = MagicMock(side_effect=["Step 1", "Step 2"])
        updates = TelegramUpdates(self.config, snapshot)
        waits = self.run_periods(updates)
        self.assertEqual(waits, [2, 2, 2])
        calls = self.opener.open.call_args_list
        self.assertEqual(len(calls), 2)
        for index, call in enumerate(calls, 1):
            request = call.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url,
                             "https://api.telegram.org/bot123:PRIVATE_TEST_TOKEN/sendMessage")
            self.assertEqual(json.loads(request.data), {
                "chat_id": -12345, "text": f"test trainer Ω\nStep {index}",
            })
            self.assertEqual(call.kwargs["timeout"], 5.0)

    def test_failures_are_redacted_and_retried_next_period(self):
        self.opener.open.side_effect = OSError("https://api.telegram.org/botPRIVATE_TEST_TOKEN")
        updates = TelegramUpdates(self.config, lambda: "Step 1")
        with self.assertLogs("LRAEP_trainer", level="WARNING") as logs:
            self.run_periods(updates)
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertNotIn("PRIVATE_TEST_TOKEN", " ".join(logs.output))

    def test_api_rejection_does_not_stop_the_next_report(self):
        self.opener.open.return_value.__enter__.return_value.read.return_value = b'{"ok":false}'
        with self.assertLogs("LRAEP_trainer", level="WARNING") as logs:
            self.run_periods(TelegramUpdates(self.config, lambda: "Step 1"))
        self.assertEqual(len(logs.output), 2)
        self.assertEqual(self.opener.open.call_count, 2)

    def test_disabled_or_closed_reporter_never_opens_a_connection(self):
        with patch("LRAEP_trainer.telegram.http_opener") as opener:
            with TelegramUpdates({}, lambda: "unused") as disabled:
                self.assertIsNone(disabled._thread)
            enabled = TelegramUpdates(self.config, lambda: "unused")
            enabled.close()
            enabled.start()
            self.assertIsNone(enabled._thread)
        opener.assert_not_called()

    def test_worker_keeps_the_callers_vpn_route(self):
        proxy = SocksProxy("127.0.0.1", 19080)
        with routed(proxy):
            updates = TelegramUpdates(self.config, lambda: "Step 1")
        with patch("LRAEP_trainer.telegram.http_opener", return_value=self.opener) as opener:
            updates._stopped.set()
            updates._run()
        opener.assert_called_once_with(proxy)

    def test_network_work_runs_in_background_and_close_stops_reports(self):
        self.config["telegram"]["update_interval_seconds"] = 0.01
        sent = threading.Event()
        threads = []
        caller = threading.get_ident()

        def snapshot():
            threads.append(threading.get_ident())
            sent.set()
            return "Step 1"

        with patch("LRAEP_trainer.telegram.http_opener", return_value=self.opener):
            updates = TelegramUpdates(self.config, snapshot).start()
            try:
                self.assertTrue(sent.wait(1))
            finally:
                updates.close()
            self.assertFalse(updates._thread.is_alive())
        self.assertTrue(all(thread != caller for thread in threads))


if __name__ == "__main__":
    unittest.main()
