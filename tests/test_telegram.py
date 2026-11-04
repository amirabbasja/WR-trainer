import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from workrunner.config import ConfigError, DEFAULT_CONFIG, load_config, save_config
from workrunner.network import SocksProxy, routed
from workrunner.telegram import TelegramUpdates


def configuration():
    value = copy.deepcopy(DEFAULT_CONFIG)
    value["trainer_name"] = "test trainer Ω"
    value["telegram"] = {
        "enabled": True, "tgBot_key": "123:PRIVATE_TEST_TOKEN", "tgChatID": -12345,
        "training_stats_tgBot_key": "456:PRIVATE_TRAINING_STATS_TOKEN",
        "update_interval_seconds": 2,
    }
    value["vpn"]["link"] = "vless://fixture@example.invalid:443"
    return value


class TelegramConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.json"
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
            ("tgBot_key", ""), ("training_stats_tgBot_key", "PRIVATE_TEST_TOKEN"),
            ("training_stats_tgBot_key", True), ("tgChatID", ""), ("tgChatID", True),
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
                updates._stopped.set()
            else:
                clock[0] += seconds
            return False

        with patch.object(updates, "_wake") as wake, \
                patch("workrunner.telegram.time.monotonic", side_effect=lambda: clock[0]), \
                patch("workrunner.telegram.http_opener", return_value=self.opener):
            wake.wait.side_effect = wait
            updates._run()
        return waits

    def test_periodic_payload_matches_progress_and_uses_training_stats_bot(self):
        snapshot = MagicMock(side_effect=['1 | job-0 | 80% | 00:01:00 | 123.45',
                                          '2 | job-1 | 50% | 00:00:30 | 4'])
        updates = TelegramUpdates(self.config, snapshot)
        waits = self.run_periods(updates)
        self.assertEqual(waits, [2, 2, 2])
        calls = self.opener.open.call_args_list
        self.assertEqual(len(calls), 2)
        for index, call in enumerate(calls, 1):
            request = call.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url,
                             "https://api.telegram.org/bot456:PRIVATE_TRAINING_STATS_TOKEN/sendMessage")
            self.assertEqual(json.loads(request.data), {
                "chat_id": -12345, "text": ['1 | job-0 | 80% | 00:01:00 | 123.45',
                                           '2 | job-1 | 50% | 00:00:30 | 4'][index - 1],
            })
            self.assertEqual(call.kwargs["timeout"], 5.0)

    def test_failures_are_redacted_and_retried_next_period(self):
        self.opener.open.side_effect = OSError("https://api.telegram.org/botPRIVATE_TEST_TOKEN")
        updates = TelegramUpdates(self.config, lambda: "Step 1")
        with self.assertLogs("workrunner", level="WARNING") as logs:
            self.run_periods(updates)
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertNotIn("PRIVATE_TEST_TOKEN", " ".join(logs.output))

    def test_api_rejection_does_not_stop_the_next_report(self):
        self.opener.open.return_value.__enter__.return_value.read.return_value = b'{"ok":false}'
        with self.assertLogs("workrunner", level="WARNING") as logs:
            self.run_periods(TelegramUpdates(self.config, lambda: "Step 1"))
        self.assertEqual(len(logs.output), 2)
        self.assertEqual(self.opener.open.call_count, 2)

    def test_disabled_or_closed_reporter_never_opens_a_connection(self):
        with patch("workrunner.telegram.http_opener") as opener:
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
        with patch("workrunner.telegram.http_opener", return_value=self.opener) as opener:
            updates._stopped.set()
            updates._run()
        opener.assert_called_once_with(proxy)

    def test_events_use_main_bot_and_flush_on_close(self):
        updates = TelegramUpdates(self.config, lambda: "unused")
        updates.event('Job 1 job-0 started')
        updates.event('Job 1 job-0 finished')
        with patch("workrunner.telegram.http_opener", return_value=self.opener):
            updates.start()
            updates.close()
        self.assertFalse(updates._thread.is_alive())
        calls = self.opener.open.call_args_list
        self.assertEqual(len(calls), 2)
        for call, message in zip(calls, ['Job 1 job-0 started', 'Job 1 job-0 finished']):
            self.assertEqual(call.args[0].full_url,
                             "https://api.telegram.org/bot123:PRIVATE_TEST_TOKEN/sendMessage")
            self.assertEqual(json.loads(call.args[0].data)["text"],
                             f"test trainer Ω\n{message}")

    def test_connection_loss_alert_uses_direct_telegram_route(self):
        proxy = SocksProxy("127.0.0.1", 19080)
        direct = MagicMock()
        direct.open.return_value.__enter__.return_value.read.return_value = b'{"ok":true}'
        with routed(proxy):
            updates = TelegramUpdates(self.config, lambda: "unused")
        updates.event("VPN connection lost", direct=True)
        updates.close()
        with patch("workrunner.telegram.http_opener", side_effect=[self.opener, direct]) as opener:
            updates._run()
        self.assertEqual([call.args[0] for call in opener.call_args_list], [proxy, None])
        self.opener.open.assert_not_called()
        self.assertEqual(direct.open.call_args.args[0].full_url,
                         "https://api.telegram.org/bot123:PRIVATE_TEST_TOKEN/sendMessage")

    def test_blank_training_stats_token_never_sends_progress_to_main_bot(self):
        self.config["telegram"]["training_stats_tgBot_key"] = ""
        updates = TelegramUpdates(self.config, lambda: "unused")
        updates._send(self.opener)
        self.opener.open.assert_not_called()
        updates.event("Job started")
        updates.close()
        with patch("workrunner.telegram.http_opener", return_value=self.opener):
            updates._run()
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertEqual(self.opener.open.call_args.args[0].full_url,
                         "https://api.telegram.org/bot123:PRIVATE_TEST_TOKEN/sendMessage")

    def test_closed_and_disabled_reporters_discard_new_events(self):
        disabled = TelegramUpdates({}, lambda: "unused")
        disabled.event("unused")
        self.assertFalse(disabled._events)
        enabled = TelegramUpdates(self.config, lambda: "unused")
        enabled.close()
        enabled.event("unused", direct=True)
        self.assertFalse(enabled._events)

    def test_failed_events_retry_without_exposing_credentials(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        self.opener.open.side_effect = [OSError("PRIVATE_TEST_TOKEN"), response]
        updates = TelegramUpdates(self.config, lambda: "progress")
        updates._training_stats_token = ""
        updates.event("Job started")
        with self.assertLogs("workrunner", level="WARNING") as logs:
            waits = self.run_periods(updates, count=1)
        self.assertEqual(waits, [2, None])
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertNotIn("PRIVATE_TEST_TOKEN", " ".join(logs.output))

    def test_urgent_loss_event_bypasses_an_older_event_retry_delay(self):
        self.config["telegram"]["training_stats_tgBot_key"] = ""
        updates = TelegramUpdates(self.config, lambda: "unused")
        deliveries = []
        waits = []
        routed_opener = MagicMock()
        direct_opener = MagicMock()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'

        def send_routed(request, **kwargs):
            deliveries.append(("routed", json.loads(request.data)["text"]))
            if len(deliveries) == 1:
                raise OSError("route unavailable")
            return response

        def send_direct(request, **kwargs):
            deliveries.append(("direct", json.loads(request.data)["text"]))
            return response

        def wait(seconds):
            waits.append(seconds)
            if len(waits) == 1:
                updates.event("VPN lost", direct=True)
            else:
                updates._stopped.set()

        routed_opener.open.side_effect = send_routed
        direct_opener.open.side_effect = send_direct
        updates.event("Job started")
        with patch("workrunner.telegram.time.monotonic", return_value=0), \
                patch.object(updates, "_wake") as wake, \
                patch("workrunner.telegram.http_opener", side_effect=[routed_opener, direct_opener]), \
                self.assertLogs("workrunner", level="WARNING"):
            wake.wait.side_effect = wait
            updates._run()
        self.assertEqual([route for route, _ in deliveries], ["routed", "direct", "routed"])
        self.assertIn("VPN lost", deliveries[1][1])
        self.assertEqual(waits, [2, 2])

    def test_close_delivery_budget_prevents_starting_an_expired_request(self):
        updates = TelegramUpdates(self.config, lambda: "unused")
        updates._close_deadline = 10
        with patch("workrunner.telegram.time.monotonic", return_value=10):
            with self.assertRaises(TimeoutError):
                updates._request(self.opener, "123:test", "Job finished")
        self.opener.open.assert_not_called()

    def test_network_work_runs_in_background_and_close_stops_reports(self):
        self.config["telegram"]["update_interval_seconds"] = 0.01
        sent = threading.Event()
        threads = []
        caller = threading.get_ident()

        def snapshot():
            threads.append(threading.get_ident())
            sent.set()
            return "Step 1"

        with patch("workrunner.telegram.http_opener", return_value=self.opener):
            updates = TelegramUpdates(self.config, snapshot).start()
            try:
                self.assertTrue(sent.wait(1))
            finally:
                updates.close()
            self.assertFalse(updates._thread.is_alive())
        self.assertTrue(all(thread != caller for thread in threads))


if __name__ == "__main__":
    unittest.main()
