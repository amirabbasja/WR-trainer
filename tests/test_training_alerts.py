import atexit
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {
        "trainer_name": "alert test trainer",
        "telegram": {"enabled": True, "tgBot_key": "123:TEST_TOKEN", "tgChatID": -12345,
                     "training_stats_tgBot_key": "456:UNUSED_STATS_TOKEN"},
        "training": {"generate_artifacts": False,
                     "job_sequence": [{"max_gpu_utilization": 80}]},
        "S3": {"protocol": "WR"},
        "vpn": {"enabled": True, "link": "vless://fixture@example.invalid:443"},
        "side_job": {},
    }


class TrainingFailureAlertTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("training_alert_entry", ROOT / "train.py")
        self.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entry)
        self.addCleanup(atexit.unregister, self.entry.stop_training)
        self.config = configuration()
        self.opener = MagicMock()
        self.opener.open.return_value.__enter__.return_value.read.return_value = b'{"ok":true}'
        self.engine = types.ModuleType("workrunner.S3_training")
        self.engine.validate_S3_endpoint = MagicMock()
        self.engine.run_S3_training = MagicMock(return_value=0)

    def run_training(self, *, load_error=None):
        with patch.dict(sys.modules, {"workrunner.S3_training": self.engine}), \
                patch("workrunner.paths.set_checkout_directories"), \
                patch("workrunner.login_config.load_training_config", return_value=self.config,
                      side_effect=load_error), \
                patch("workrunner.config.resolve_S3_settings", side_effect=lambda value: value), \
                patch("workrunner.progress.in_notebook", return_value=False), \
                patch("workrunner.vpn.ManagedVPN", side_effect=lambda _: nullcontext(None)), \
                patch("workrunner.telegram.build_opener", return_value=self.opener), \
                redirect_stdout(io.StringIO()):
            return self.entry.run(ROOT)

    def assert_alert(self, message):
        self.assertEqual(self.opener.open.call_count, 1)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.telegram.org/bot123:TEST_TOKEN/sendMessage")
        self.assertEqual(json.loads(request.data), {
            "chat_id": -12345, "text": f"alert test trainer\n{message}",
        })

    def test_startup_validation_error_sends_one_alert(self):
        self.engine.validate_S3_endpoint.side_effect = ValueError("Invalid endpoint")
        self.assertEqual(self.run_training(), 1)
        self.assert_alert("Training failed (ValueError): Invalid endpoint")

    def test_unexpected_training_exception_sends_one_alert(self):
        self.engine.run_S3_training.side_effect = KeyError("missing job")
        self.assertEqual(self.run_training(), 1)
        self.assert_alert("Training failed (KeyError): 'missing job'")

    def test_normal_completion_and_keyboard_interrupt_send_no_alerts(self):
        self.assertEqual(self.run_training(), 0)
        self.opener.open.assert_not_called()
        self.engine.run_S3_training.side_effect = KeyboardInterrupt
        self.assertEqual(self.run_training(), 130)
        self.opener.open.assert_not_called()

    def test_disabled_telegram_sends_no_alert_on_failure(self):
        self.config["telegram"]["enabled"] = False
        self.engine.run_S3_training.side_effect = RuntimeError("CUDA failed")
        self.assertEqual(self.run_training(), 1)
        self.opener.open.assert_not_called()

    def test_config_validation_error_recovers_bot_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(self.config))
            with patch("workrunner.login_config.load_training_config",
                       side_effect=ValueError("Invalid training config")), \
                    patch("workrunner.paths.set_checkout_directories"), \
                    patch("workrunner.progress.in_notebook", return_value=False), \
                    patch("workrunner.telegram.build_opener", return_value=self.opener), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(self.entry.run(root), 1)
        self.assert_alert("Training failed (ValueError): Invalid training config")

    def test_standalone_side_job_setup_failure_sends_one_alert(self):
        from workrunner import aux_training
        with patch("workrunner.login_config.load_training_config", return_value=self.config), \
                patch.object(aux_training, "SideJob", side_effect=RuntimeError("side job failed")), \
                patch("workrunner.progress.in_notebook", return_value=False), \
                patch("workrunner.telegram.build_opener", return_value=self.opener):
            self.assertEqual(aux_training.main(["--config", "unused.json"]), 1)
        self.assert_alert("Training failed (RuntimeError): side job failed")

    def test_standalone_side_job_completion_sends_no_routine_events(self):
        from workrunner import aux_training
        job = MagicMock(devices=[])
        job.step.return_value = False
        job._progress_epoch = 1
        with patch("workrunner.login_config.load_training_config", return_value=self.config), \
                patch.object(aux_training, "SideJob", return_value=job), \
                patch("workrunner.progress.in_notebook", return_value=False), \
                patch("workrunner.telegram.build_opener", return_value=self.opener):
            self.assertEqual(aux_training.main(["--config", "unused.json"]), 0)
        self.opener.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
