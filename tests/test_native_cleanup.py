from contextlib import contextmanager, nullcontext
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from workrunner import native_cuda
from workrunner.stream_digest import StreamDigest as Digest
from workrunner.protocol import load_official_helpers


ROOT_FUNCTION = 'tree_root'
MATCH_FUNCTION = 'match_candidates'
HELPER_DIRECTORY = 'work_helpers'


class _Stream:
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class _DeviceWords:
    def __init__(self, values, *, record_error=None, wipe_error=None):
        self.values = values.copy()
        self.record_error = record_error
        self.wipe_error = wipe_error
        self.zeroed = False

    def record_stream(self, stream):
        if self.record_error is not None:
            raise self.record_error

    def zero_(self):
        if self.wipe_error is not None:
            raise self.wipe_error
        self.values.fill(0)
        self.zeroed = True
        return self


class _HostWords:
    def __init__(self, values, copies, *, transfer_error=None, record_error=None):
        self.values = values
        self.copies = copies
        self.transfer_error = transfer_error
        self.record_error = record_error

    def to(self, destination, **options):
        if self.transfer_error is not None:
            raise self.transfer_error
        result = _DeviceWords(self.values, record_error=self.record_error)
        self.copies.append(result)
        return result

    def zero_(self):
        self.values.fill(0)
        return self


class NativeHostCleanupTests(unittest.TestCase):
    def check_host_staging(self, *, transfer_error=None, record_error=None):
        views, copies = [], []
        stream = _Stream()

        def from_numpy(values):
            views.append(values)
            return _HostWords(values, copies, transfer_error=transfer_error, record_error=record_error)

        original = bytes(range(32))
        with patch.object(native_cuda.torch, 'from_numpy', side_effect=from_numpy), \
                patch.object(native_cuda.torch.cuda, 'current_stream', return_value=stream), \
                patch.object(native_cuda.torch.cuda, 'stream', return_value=nullcontext()):
            error = transfer_error or record_error
            if error is None:
                result = native_cuda._words(original, 'cuda:0')
                np.testing.assert_array_equal(result.values, np.frombuffer(original, dtype='<i4'))
            else:
                with self.assertRaises(type(error)) as caught:
                    native_cuda._words(original, 'cuda:0')
                self.assertIs(caught.exception, error)
        self.assertTrue(views)
        self.assertTrue(all(not np.any(view) for view in views))
        self.assertEqual(original, bytes(range(32)))
        return copies, stream

    def test_cpu_staging_is_overwritten_after_successful_transfer(self):
        self.check_host_staging()

    def test_cpu_staging_is_overwritten_when_transfer_fails(self):
        self.check_host_staging(transfer_error=RuntimeError('transfer failed'))

    def test_gpu_copy_is_overwritten_if_stream_registration_fails(self):
        copies, stream = self.check_host_staging(record_error=RuntimeError('registration failed'))
        self.assertEqual(len(copies), 1)
        self.assertTrue(copies[0].zeroed)
        self.assertGreaterEqual(stream.synchronizations, 1)

    def test_cleanup_attempts_remaining_buffers_and_raises_on_failure(self):
        first = _DeviceWords(np.ones(8, dtype=np.int32), wipe_error=RuntimeError('wipe failed'))
        second = _DeviceWords(np.ones(8, dtype=np.int32))
        stream = _Stream()
        with patch.object(native_cuda.torch.cuda, 'stream', return_value=nullcontext()):
            with self.assertRaises(RuntimeError):
                native_cuda._wipe_words([first, second], stream)
        self.assertTrue(second.zeroed)
        self.assertGreaterEqual(stream.synchronizations, 1)

    def test_cleanup_failure_preserves_an_existing_error(self):
        original = ValueError('operation failed')
        buffer = _DeviceWords(np.ones(8, dtype=np.int32), wipe_error=RuntimeError('wipe failed'))
        stream = _Stream()
        with patch.object(native_cuda.torch.cuda, 'stream', return_value=nullcontext()):
            native_cuda._wipe_words([buffer], stream, original_error=original)
        self.assertTrue(getattr(original, '__notes__', ()))
        self.assertGreaterEqual(stream.synchronizations, 1)

    def test_synchronization_failure_is_reported_after_zeroing(self):
        class FailedStream(_Stream):
            def synchronize(self):
                super().synchronize()
                raise RuntimeError('synchronization failed')

        for original in (None, ValueError('operation failed')):
            with self.subTest(original_error=original):
                stream = FailedStream()
                buffer = _DeviceWords(np.ones(8, dtype=np.int32))
                with patch.object(native_cuda.torch.cuda, 'stream', return_value=nullcontext()):
                    if original is None:
                        with self.assertRaises(RuntimeError):
                            native_cuda._wipe_words([buffer], stream)
                    else:
                        native_cuda._wipe_words([buffer], stream, original_error=original)
                        self.assertTrue(getattr(original, '__notes__', ()))
                self.assertTrue(buffer.zeroed)
                self.assertEqual(stream.synchronizations, 1)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
class NativeDeviceCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_cuda.load_native('cuda:0')
        load_official_helpers(Path(__file__).resolve().parents[1] / 'src' / HELPER_DIRECTORY)
        from S5_base.noise_generation import NoiseGenerator
        cls.noise_generator = NoiseGenerator(noise_rank=128)
        cls.device = torch.device('cuda:0')

    @contextmanager
    def capture_words(self):
        saved = []
        allocate = native_cuda._words

        def remember(*args, **options):
            result = allocate(*args, **options)
            saved.append(result)
            return result

        with patch.object(native_cuda, '_words', side_effect=remember):
            yield saved

    def assert_wiped(self, tensors, count):
        self.assertEqual(len(tensors), count)
        for tensor in tensors:
            self.assertEqual(tensor.numel(), 8)
            self.assertTrue(bool(torch.all(tensor.cpu() == 0)))

    def test_commitment_key_is_overwritten_and_result_is_unchanged(self):
        key = bytes(range(32))
        signal_cpu = torch.arange(511, dtype=torch.int32).to(torch.int8).reshape(1, -1)
        expected = Digest(signal_cpu.numpy().tobytes().ljust(1024, b'\0'), key=key).digest()
        signal = signal_cpu.to(self.device)
        with self.capture_words() as saved:
            actual = getattr(native_cuda, ROOT_FUNCTION)(signal, key)
        self.assertEqual(actual, expected)
        self.assert_wiped(saved, 1)
        self.assertTrue(torch.equal(signal.cpu(), signal_cpu))
        self.assertEqual(key, bytes(range(32)))

    def test_noise_keys_are_overwritten_without_corrupting_outputs_on_nondefault_stream(self):
        key_a, key_b = bytes(range(32)), bytes(reversed(range(32)))
        a_cpu = torch.full((128, 128), 7, dtype=torch.int8)
        bt_cpu = torch.full((128, 128), -5, dtype=torch.int8)
        al, ar, bl, br = self.noise_generator.generate_noise_metrices(key_a, key_b, 128, 128, 128)
        expected_a = a_cpu + (al.to(torch.int32) @ ar.to(torch.int32)).to(torch.int8)
        expected_bt = bt_cpu + (bl.to(torch.int32) @ br.to(torch.int32)).T.to(torch.int8)
        stream = torch.cuda.Stream(device=self.device)
        with self.capture_words() as saved, torch.cuda.stream(stream):
            a, bt = a_cpu.to(self.device), bt_cpu.to(self.device)
            actual_a, actual_bt = native_cuda.add_noise(a, bt, key_a, key_b)
        self.assert_wiped(saved, 2)
        self.assertTrue(torch.equal(actual_a.cpu(), expected_a))
        self.assertTrue(torch.equal(actual_bt.cpu(), expected_bt))
        self.assertTrue(torch.equal(a.cpu(), a_cpu))
        self.assertTrue(torch.equal(bt.cpu(), bt_cpu))

    def test_candidate_keys_and_target_are_overwritten_after_capacity_retries(self):
        key = bytes(range(32))
        training_threshold = (1 << 256) - 1
        words = np.arange(3 * 5 * 16, dtype=np.uint32).reshape(3, 5, 16)
        expected = [
            (row, col, Digest(words[row, col].astype('<u4').tobytes(), key=key).digest())
            for row in range(3) for col in range(5)
        ]
        stream = torch.cuda.Stream(device=self.device)
        with self.capture_words() as saved, torch.cuda.stream(stream):
            transcripts = torch.from_numpy(words.view(np.int32)).to(self.device)
            actual = getattr(native_cuda, MATCH_FUNCTION)(transcripts, key, training_threshold, capacity=1)
        self.assertEqual(actual, expected)
        self.assert_wiped(saved, 2)
        np.testing.assert_array_equal(transcripts.cpu().numpy().view(np.uint32), words)
        self.assertEqual(key, bytes(range(32)))
        self.assertEqual(training_threshold, (1 << 256) - 1)

    def test_candidate_no_match_return_still_overwrites_inputs(self):
        key = bytes(range(32))
        transcripts = torch.zeros((1, 1, 16), dtype=torch.int32, device=self.device)
        self.assertNotEqual(Digest(bytes(64), key=key).digest(), bytes(32))
        with self.capture_words() as saved:
            actual = getattr(native_cuda, MATCH_FUNCTION)(transcripts, key, 0)
        self.assertEqual(actual, [])
        self.assert_wiped(saved, 2)

    def test_public_operations_overwrite_copies_after_kernel_errors(self):
        key = bytes(range(32))
        signal = torch.ones((128, 128), dtype=torch.int8, device=self.device)
        transcripts = torch.ones((1, 1, 16), dtype=torch.int32, device=self.device)
        operations = (
            (lambda: getattr(native_cuda, ROOT_FUNCTION)(signal, key), 1),
            (lambda: native_cuda.add_noise(signal, signal, key, key), 2),
            (lambda: getattr(native_cuda, MATCH_FUNCTION)(transcripts, key, (1 << 256) - 1), 2),
        )
        for operation, count in operations:
            with self.subTest(operation=operation), self.capture_words() as saved:
                error = RuntimeError('injected operation failure')
                with patch.object(native_cuda, '_check', side_effect=error):
                    with self.assertRaises(RuntimeError) as caught:
                        operation()
                self.assertIs(caught.exception, error)
                self.assert_wiped(saved, count)

    def test_partial_key_allocation_failure_overwrites_the_first_copy(self):
        key = bytes(range(32))
        signal = torch.ones((128, 128), dtype=torch.int8, device=self.device)
        allocate = native_cuda._words
        saved = []
        error = MemoryError('injected second allocation failure')

        def remember(value, device):
            if saved:
                raise error
            result = allocate(value, device)
            saved.append(result)
            return result

        with patch.object(native_cuda, '_words', side_effect=remember):
            with self.assertRaises(MemoryError) as caught:
                native_cuda.add_noise(signal, signal, key, key)
        self.assertIs(caught.exception, error)
        self.assert_wiped(saved, 1)


if __name__ == '__main__':
    unittest.main()
