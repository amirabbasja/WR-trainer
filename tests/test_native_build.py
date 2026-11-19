from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from workrunner import native_cuda
from workrunner.training_artifacts import training_output


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / 'cache'
        self.builds = []
        self.library = object()
        patches = [
            patch.object(native_cuda, '_LIBRARIES', {}),
            patch.object(native_cuda, 'cache_dir', return_value=self.cache),
            patch.object(native_cuda.torch.cuda, 'is_available', return_value=True),
            patch.object(native_cuda.torch.cuda, 'get_device_capability', return_value=(7, 5)),
            patch.object(native_cuda.shutil, 'which', return_value='/cuda/bin/nvcc'),
            patch.object(native_cuda.ctypes, 'CDLL', side_effect=self.load_library),
            patch.object(native_cuda, '_configure', return_value=self.library),
            patch.object(native_cuda.subprocess, 'run', side_effect=self.compile_library),
        ]
        for mock_patch in patches:
            mock_patch.start()
            self.addCleanup(mock_patch.stop)

    def compile_library(self, command, **kwargs):
        output = Path(command[-1])
        self.builds.append(output)
        output.write_bytes(b'test library')
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

    def load_library(self, path):
        self.assertTrue(Path(path).is_file(), 'library must exist until dlopen finishes')
        return self.library

    def test_fresh_quiet_run_builds_temporarily_and_reuses_loaded_library(self):
        with training_output(False):
            self.assertIs(native_cuda.load_native(), self.library)
            self.assertIs(native_cuda.load_native(), self.library)
        self.assertEqual(len(self.builds), 1)
        self.assertFalse(self.builds[0].parent.exists())
        self.assertFalse(self.cache.exists())

    def test_artifact_enabled_build_is_cached_for_later_quiet_runs(self):
        self.assertIs(native_cuda.load_native(), self.library)
        self.assertEqual(len(self.builds), 1)
        build_directory = self.builds[0].parent
        self.assertTrue((build_directory / 'build.log').is_file())
        self.assertEqual(len(list(build_directory.glob('*.so'))), 1)
        native_cuda._LIBRARIES.clear()
        with training_output(False):
            self.assertIs(native_cuda.load_native(), self.library)
        self.assertEqual(len(self.builds), 1)

    def test_quiet_build_failure_preserves_diagnostic_and_cleans_up(self):
        def fail(command, **kwargs):
            self.compile_library(command, **kwargs)
            return subprocess.CompletedProcess(command, 1, stdout='', stderr='unsupported GPU architecture')

        with patch.object(native_cuda.subprocess, 'run', side_effect=fail):
            with training_output(False):
                with self.assertRaisesRegex(native_cuda.NativeUnavailable, 'unsupported GPU architecture'):
                    native_cuda.load_native()
        self.assertFalse(self.builds[0].parent.exists())
        self.assertFalse(self.cache.exists())
        self.assertEqual(native_cuda._LIBRARIES, {})

    def test_quiet_build_timeout_cleans_up(self):
        def timeout(command, **kwargs):
            self.compile_library(command, **kwargs)
            raise subprocess.TimeoutExpired(command, 180)

        with patch.object(native_cuda.subprocess, 'run', side_effect=timeout):
            with training_output(False):
                with self.assertRaises(native_cuda.NativeUnavailable):
                    native_cuda.load_native()
        self.assertFalse(self.builds[0].parent.exists())
        self.assertFalse(self.cache.exists())


if __name__ == '__main__':
    unittest.main()
