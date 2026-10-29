

from contextlib import contextmanager, redirect_stdout
import ctypes
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class TrainingControlTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("wr_control_test", ROOT / "train.py")
        self.trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.trainer)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.trainer.stop_training)
        self.ready = Path(self.temporary.name) / "ready"

    def start_script(self, script, **options):

        original_popen = subprocess.Popen

        def launch(_command, **kwargs):
            return original_popen([sys.executable, "-u", "-c", script], **kwargs)

        with patch.object(self.trainer.subprocess, "Popen", side_effect=launch):
            return self.trainer.start(ROOT, **options)

    def wait_ready(self, process, timeout=10, *, path=None):
        path = self.ready if path is None else path
        deadline = time.monotonic() + timeout
        while not path.exists():
            self.assertIsNone(process.poll(), "test worker exited before becoming ready")
            if time.monotonic() >= deadline:
                self.fail("test worker did not become ready")
            time.sleep(0.01)

    def sleeper(self, *, ignore_term=False):
        handler = "signal.SIG_IGN" if ignore_term else "lambda *_: sys.exit(0)"
        return (
            "import signal, sys, time\n"
            "from pathlib import Path\n"
            f"signal.signal(signal.SIGTERM, {handler})\n"
            f"Path({str(self.ready)!r}).write_text('ready')\n"
            "while True:\n"
            "    time.sleep(0.05)\n"
        )

    def test_stop_without_worker_is_safe_and_repeatable(self):
        self.assertIsNone(self.trainer.status())
        self.assertIsNone(self.trainer.stop_training())
        self.assertIsNone(self.trainer.stop_training())

    @unittest.skipUnless(os.name == "posix", "SIGTERM handlers require POSIX")
    def test_stop_waits_for_worker_to_exit_and_can_be_repeated(self):
        process = self.start_script(self.sleeper())
        self.wait_ready(process)
        self.assertIsNone(self.trainer.status())
        self.assertEqual(self.trainer.stop_training(), 0)
        self.assertEqual(process.poll(), 0)
        self.assertEqual(self.trainer.status(), 0)
        self.assertIsNone(self.trainer.stop_training())
        self.assertEqual(self.trainer.status(), 0)
        self.assertEqual(process.poll(), 0)

    def test_start_rejects_a_second_running_worker(self):
        process = self.start_script(self.sleeper())
        self.wait_ready(process)
        with patch.object(self.trainer.subprocess, "Popen") as launch:
            with self.assertRaises(RuntimeError):
                self.trainer.start(ROOT)
        launch.assert_not_called()
        self.assertIsNone(process.poll())

    def test_status_preserves_success_and_failure_after_worker_exit(self):
        for code in (0, 7):
            with self.subTest(code=code):
                process = self.start_script(f"raise SystemExit({code})")
                self.assertEqual(process.wait(timeout=5), code)
                self.assertEqual(self.trainer.status(), code)
                self.assertEqual(self.trainer.stop_training(), code)
                self.assertEqual(self.trainer.status(), code)

    @unittest.skipUnless(os.name == "posix", "process-group kill requires POSIX")
    def test_stop_forces_exit_after_grace_period(self):
        process = self.start_script(self.sleeper(ignore_term=True))
        self.wait_ready(process)
        self.assertEqual(self.trainer.stop_training(timeout=0.05), -signal.SIGKILL)
        self.assertEqual(process.poll(), -signal.SIGKILL)

    @contextmanager
    def adopt_test_descendant(self, pid_file):

        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        try:
            yield
        finally:
            if pid_file.exists():
                pid = int(pid_file.read_text())
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
            libc.prctl(36, previous.value, 0, 0, 0)

    def wait_dead(self, pid):
        deadline = time.monotonic() + 5
        while True:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                return
            if state == "Z":
                return
            if time.monotonic() >= deadline:
                self.fail("worker descendant survived shutdown")
            time.sleep(0.01)

    def descendant_script(self, pid_file, *, natural_exit=False):
        child_ready = Path(self.temporary.name) / "child-ready"
        child = (
            "import signal,time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            f"Path({str(child_ready)!r}).write_text('ready')\n"
            "time.sleep(60)\n"
        )
        return (
            "import signal, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            f"Path({str(pid_file)!r}).write_text(str(child.pid))\n"
            f"while not Path({str(child_ready)!r}).exists(): time.sleep(0.01)\n"
            f"Path({str(self.ready)!r}).write_text('ready')\n"
            + ("raise SystemExit(7)\n" if natural_exit else "time.sleep(60)\n")
        )

    @unittest.skipUnless(sys.platform == "linux", "process adoption requires Linux")
    def test_stop_kills_worker_descendants_too(self):
        descendant = Path(self.temporary.name) / "descendant"
        with self.adopt_test_descendant(descendant):
            process = self.start_script(self.descendant_script(descendant))
            self.wait_ready(process)
            child_pid = int(descendant.read_text())
            self.trainer.stop_training(timeout=0.05)
            self.assertIsNotNone(process.poll())
            self.wait_dead(child_pid)

    @unittest.skipUnless(sys.platform == "linux", "process adoption requires Linux")
    def test_natural_worker_exit_automatically_cleans_up_descendant(self):
        descendant = Path(self.temporary.name) / "descendant"
        with self.adopt_test_descendant(descendant):
            process = self.start_script(self.descendant_script(descendant, natural_exit=True))
            self.assertEqual(process.wait(timeout=5), 7)
            child_pid = int(descendant.read_text())

            self.wait_dead(child_pid)
            self.assertEqual(self.trainer.status(), 7)

    @unittest.skipUnless(sys.platform == "linux", "parent adoption requires Linux")
    def test_worker_stops_itself_after_its_parent_is_killed(self):
        descendant = Path(self.temporary.name) / "orphan-worker"
        worker = (
            "import importlib.util, sys, threading, time\n"
            "from pathlib import Path\n"
            f"spec = importlib.util.spec_from_file_location('worker_entry', {str(ROOT / 'train.py')!r})\n"
            "entry = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(entry)\n"
            "threading.Thread(target=entry._watch_parent, args=(int(sys.argv[1]),), daemon=True).start()\n"
            f"Path({str(self.ready)!r}).write_text('ready')\n"
            "time.sleep(60)\n"
        )
        parent_script = (
            "import os, signal, subprocess, sys, time\n"
            "from pathlib import Path\n"
            f"child = subprocess.Popen([sys.executable, '-c', {worker!r}, str(os.getpid())], "
            "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            f"Path({str(descendant)!r}).write_text(str(child.pid))\n"
            f"while not Path({str(self.ready)!r}).exists():\n"
            "    if child.poll() is not None: raise SystemExit(2)\n"
            "    time.sleep(0.01)\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n"
        )
        with self.adopt_test_descendant(descendant):
            parent = subprocess.Popen([sys.executable, '-c', parent_script])
            try:
                self.assertEqual(parent.wait(timeout=5), -signal.SIGKILL)
                self.wait_dead(int(descendant.read_text()))
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=5)

    def test_closed_notebook_output_is_drained_until_explicit_stop(self):
        go = Path(self.temporary.name) / "go"
        drained = Path(self.temporary.name) / "drained"
        script = (
            "import signal, sys, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
            f"Path({str(self.ready)!r}).write_text('ready')\n"
            f"while not Path({str(go)!r}).exists(): time.sleep(0.01)\n"
            "for _ in range(128): print('x' * 4096, flush=True)\n"
            f"Path({str(drained)!r}).write_text('drained')\n"
            "time.sleep(60)\n"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            process = self.start_script(script)
            self.wait_ready(process)
            output.close()
            go.write_text("go")
        self.wait_ready(process, path=drained)
        self.assertIsNone(process.poll())
        self.assertEqual(self.trainer.stop_training(), 0)

    @unittest.skipUnless(
        os.name == "posix" and importlib.util.find_spec("torch"),
        "CUDA allocation check requires POSIX and PyTorch",
    )
    def test_stop_reaps_worker_holding_cuda_allocations(self):
        script = (
            "import signal, sys, time, torch\n"
            "from pathlib import Path\n"
            "if not torch.cuda.is_available():\n"
            "    raise SystemExit(77)\n"
            "tensor = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device='cuda')\n"
            "tensor.fill_(1)\n"
            "torch.cuda.synchronize()\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
            f"Path({str(self.ready)!r}).write_text(str(torch.cuda.memory_allocated()))\n"
            "while True:\n"
            "    time.sleep(0.05)\n"
        )
        process = self.start_script(script)
        deadline = time.monotonic() + 30
        while not self.ready.exists():
            code = process.poll()
            if code == 77:
                self.skipTest("CUDA is unavailable in the worker")
            self.assertIsNone(code, "CUDA test worker failed to initialize")
            if time.monotonic() >= deadline:
                self.fail("CUDA test worker did not become ready")
            time.sleep(0.02)
        self.assertGreaterEqual(int(self.ready.read_text()), 64 * 1024 * 1024)
        self.assertEqual(self.trainer.stop_training(), 0)
        self.assertEqual(process.poll(), 0)
        with self.assertRaises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)


if __name__ == "__main__":
    unittest.main()
