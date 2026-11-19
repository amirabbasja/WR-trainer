import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from workrunner.config import ConfigError, DEFAULT_CONFIG, load_config, save_config
from workrunner.socks_proxy import SocksProxy, routed
from workrunner.telegram import TelegramUpdates, report_training_error
from workrunner.training_artifacts import training_output


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
                clock[0] += 10 if seconds is None else seconds
            return False

        with patch.object(updates, "_wake") as wake, \
                patch("workrunner.telegram.time.monotonic", side_effect=lambda: clock[0]), \
                patch("workrunner.telegram.build_opener", return_value=self.opener), \
                patch("workrunner.telegram.http_opener", return_value=self.opener):
            wake.wait.side_effect = wait
            updates._run()
        return waits

    def test_idle_reporter_never_sends_progress_even_with_stats_bot_configured(self):
        updates = TelegramUpdates(self.config)
        self.assertEqual(self.run_periods(updates), [None, None, None])
        self.opener.open.assert_not_called()

    def test_api_rejection_does_not_stop_later_alerts(self):
        self.opener.open.return_value.__enter__.return_value.read.return_value = b'{"ok":false}'
        updates = TelegramUpdates(self.config)
        updates.event("Training failed (RuntimeError): CUDA failure")
        with self.assertLogs("workrunner", level="WARNING"):
            self.run_periods(updates, count=1)
        self.assertGreaterEqual(self.opener.open.call_count, 2)

    def test_disabled_or_closed_reporter_never_opens_a_connection(self):
        with patch("workrunner.telegram.http_opener") as opener:
            with TelegramUpdates({}) as disabled:
                self.assertIsNone(disabled._thread)
            enabled = TelegramUpdates(self.config)
            enabled.close()
            enabled.start()
            self.assertIsNone(enabled._thread)
        opener.assert_not_called()

    def test_worker_keeps_the_callers_vpn_route(self):
        proxy = SocksProxy("127.0.0.1", 19080)
        with routed(proxy):
            updates = TelegramUpdates(self.config)
        updates.event("Training failed: route failure")
        updates.close()
        with patch("workrunner.telegram.http_opener", return_value=self.opener) as opener:
            updates._run()
        opener.assert_called_once_with(proxy)

    def test_events_use_main_bot_and_flush_on_close(self):
        updates = TelegramUpdates(self.config)
        updates.event('Training failed (RuntimeError): CUDA failure')
        updates.event('Training failed (OSError): connection failed')
        with patch("workrunner.telegram.build_opener", return_value=self.opener), \
                patch("workrunner.telegram.http_opener", return_value=self.opener):
            updates.start()
            updates.close()
        self.assertFalse(updates._thread.is_alive())
        calls = self.opener.open.call_args_list
        self.assertEqual(len(calls), 2)
        for call, message in zip(calls, ['Training failed (RuntimeError): CUDA failure', 'Training failed (OSError): connection failed']):
            self.assertEqual(call.args[0].full_url,
                             "https://api.telegram.org/bot123:PRIVATE_TEST_TOKEN/sendMessage")
            self.assertEqual(json.loads(call.args[0].data)["text"],
                             f"test trainer Ω\n{message}")

    def test_connection_loss_alert_uses_direct_telegram_route(self):
        proxy = SocksProxy("127.0.0.1", 19080)
        direct = MagicMock()
        direct.open.return_value.__enter__.return_value.read.return_value = b'{"ok":true}'
        with routed(proxy):
            updates = TelegramUpdates(self.config)
        updates.event("VPN connection lost", direct=True)
        updates.close()
        with patch("workrunner.telegram.http_opener") as opener, \
                patch("workrunner.telegram.build_opener", return_value=direct) as direct_factory:
            updates._run()
        opener.assert_not_called()
        self.assertEqual(direct_factory.call_args.args[0].proxies, {})
        self.opener.open.assert_not_called()
        self.assertEqual(direct.open.call_args.args[0].full_url,
                         "https://api.telegram.org/bot123:PRIVATE_TEST_TOKEN/sendMessage")

    def test_error_alert_does_not_require_training_stats_bot(self):
        self.config["telegram"]["training_stats_tgBot_key"] = ""
        with patch("workrunner.telegram.build_opener", return_value=self.opener), \
                patch("workrunner.telegram.http_opener", return_value=self.opener):
            report_training_error(self.config, RuntimeError("CUDA failure"))
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertEqual(self.opener.open.call_args.args[0].full_url,
                         "https://api.telegram.org/bot123:PRIVATE_TEST_TOKEN/sendMessage")

    def test_closed_and_disabled_reporters_discard_new_events(self):
        disabled = TelegramUpdates({})
        disabled.event("unused")
        self.assertFalse(disabled._events)
        enabled = TelegramUpdates(self.config)
        enabled.close()
        enabled.event("unused", direct=True)
        self.assertFalse(enabled._events)

    def test_failed_events_retry_without_exposing_credentials(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        self.opener.open.side_effect = [OSError("PRIVATE_TEST_TOKEN"), response]
        updates = TelegramUpdates(self.config)
        updates.event("Training failed (RuntimeError): CUDA failure")
        with self.assertLogs("workrunner", level="WARNING") as logs:
            waits = self.run_periods(updates, count=1)
        self.assertEqual(waits, [2, None])
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertNotIn("PRIVATE_TEST_TOKEN", " ".join(logs.output))

    def test_urgent_loss_event_bypasses_an_older_event_retry_delay(self):
        self.config["telegram"]["training_stats_tgBot_key"] = ""
        updates = TelegramUpdates(self.config)
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
        updates.event("Training failed (RuntimeError): CUDA failure")
        with patch("workrunner.telegram.time.monotonic", return_value=0), \
                patch.object(updates, "_wake") as wake, \
                patch("workrunner.telegram.http_opener", return_value=routed_opener), \
                patch("workrunner.telegram.build_opener", return_value=direct_opener), \
                self.assertLogs("workrunner", level="WARNING"):
            wake.wait.side_effect = wait
            updates._run()
        self.assertEqual([route for route, _ in deliveries], ["routed", "direct", "routed"])
        self.assertIn("VPN lost", deliveries[1][1])
        self.assertEqual(waits, [2, 2])

    def test_close_delivery_budget_prevents_starting_an_expired_request(self):
        updates = TelegramUpdates(self.config)
        updates._close_deadline = 10
        with patch("workrunner.telegram.time.monotonic", return_value=10):
            with self.assertRaises(TimeoutError):
                updates._request(self.opener, "123:test", "Training failed (OSError): connection failed")
        self.opener.open.assert_not_called()

    def test_error_alert_runs_without_artifacts_and_flushes_before_return(self):
        caller = threading.get_ident()
        threads = []
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'

        def send(*args, **kwargs):
            threads.append(threading.get_ident())
            return response

        self.opener.open.side_effect = send
        with training_output(False), \
                patch("workrunner.telegram.build_opener", return_value=self.opener), \
                patch("workrunner.telegram.http_opener", return_value=self.opener):
            report_training_error(self.config, KeyError("missing value"))
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], caller)
        payload = json.loads(self.opener.open.call_args.args[0].data)
        self.assertEqual(payload["text"],
                         "test trainer Ω\nTraining failed (KeyError): 'missing value'")

    def test_error_alert_uses_direct_route_after_vpn_failure(self):
        with routed(SocksProxy("127.0.0.1", 19080)), \
                patch("workrunner.telegram.build_opener", return_value=self.opener), \
                patch("workrunner.telegram.http_opener", return_value=self.opener) as opener:
            report_training_error(self.config, OSError("VPN startup failed"))
        opener.assert_not_called()
        self.assertEqual(self.opener.open.call_count, 1)

    def test_real_direct_transport_does_not_require_a_vpn_proxy(self):
        # Exercise the real transport factory without making a network request.
        with routed(SocksProxy("127.0.0.1", 19080)), \
                patch.object(TelegramUpdates, "_request") as request, \
                self.assertNoLogs("workrunner", level="WARNING"):
            report_training_error(self.config, OSError("VPN is unavailable"))
        request.assert_called_once()
        self.assertEqual(request.call_args.args[1], "123:PRIVATE_TEST_TOKEN")

    def test_error_alert_redacts_bot_tokens_and_limits_message_length(self):
        message = " ".join(self.config["telegram"][key]
                           for key in ("tgBot_key", "training_stats_tgBot_key")) + "x" * 5000
        with patch("workrunner.telegram.build_opener", return_value=self.opener), \
                patch("workrunner.telegram.http_opener", return_value=self.opener):
            report_training_error(self.config, RuntimeError(message))
        payload = json.loads(self.opener.open.call_args.args[0].data)
        self.assertEqual(len(payload["text"]), 4000)
        self.assertNotIn("PRIVATE_TEST_TOKEN", payload["text"])
        self.assertNotIn("PRIVATE_TRAINING_STATS_TOKEN", payload["text"])

    def test_config_validation_failure_recovers_alert_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(self.config), encoding="utf-8")
            with patch("workrunner.telegram.build_opener", return_value=self.opener), \
                patch("workrunner.telegram.http_opener", return_value=self.opener):
                report_training_error(None, ValueError("Invalid training setting"), config_path=path)
        self.assertEqual(self.opener.open.call_count, 1)
        payload = json.loads(self.opener.open.call_args.args[0].data)
        self.assertIn("Invalid training setting", payload["text"])

    def test_disabled_error_reporting_does_not_connect(self):
        self.config["telegram"]["enabled"] = False
        with patch("workrunner.telegram.http_opener") as opener:
            report_training_error(self.config, RuntimeError("CUDA failure"))
        opener.assert_not_called()

    def test_notification_failure_does_not_raise_or_expose_credentials(self):
        with patch("workrunner.telegram.build_opener", side_effect=OSError("PRIVATE_TEST_TOKEN")), \
                self.assertLogs("workrunner", level="WARNING") as logs:
            report_training_error(self.config, RuntimeError("CUDA failure"))
        self.assertNotIn("PRIVATE_TEST_TOKEN", " ".join(logs.output))

    def test_unreadable_config_does_not_mask_original_error(self):
        with tempfile.TemporaryDirectory() as temporary, \
                self.assertLogs("workrunner", level="WARNING"):
            report_training_error(None, ValueError("bad config"),
                                  config_path=Path(temporary) / "missing.json")


if __name__ == "__main__":
    unittest.main()
