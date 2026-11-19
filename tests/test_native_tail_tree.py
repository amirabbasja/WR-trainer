"""Seed-tail tree counters, odd promotions, padding, streams, and key cleanup."""
from contextlib import contextmanager
import unittest
from unittest.mock import patch

import numpy as np
import torch

from workrunner import native_cuda
from workrunner.stream_digest import StreamDigest


class TailTreeValidationTests(unittest.TestCase):
    def test_invalid_parameters_fail_before_loading_cuda(self):
        key = bytes(32)
        invalid = [(0, 64, key), (15, 64, key), (1025, 64, key),
                   (1039, 64, key), (2**32+1, 64, key),
                   (16, -1, key), (16, 256, key),
                   (16, 64, bytes(31)), (16, 64, bytearray(32))]
        with patch.object(native_cuda, 'load_native') as load:
            for size, value, supplied_key in invalid:
                with self.subTest(size=size, value=value), self.assertRaises(ValueError):
                    native_cuda.ConstantTailTree(size, value, supplied_key, 'cuda:0')
        load.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
class NativeTailTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_cuda.load_native('cuda:0')

    @contextmanager
    def capture_words(self):
        saved = []
        allocate = native_cuda._words

        def remember(*args, **kwargs):
            result = allocate(*args, **kwargs)
            saved.append(result)
            return result

        with patch.object(native_cuda, '_words', side_effect=remember):
            yield saved

    def assert_wiped(self, buffers, count):
        self.assertEqual(len(buffers), count)
        self.assertTrue(all(buffer.numel() == 8 for buffer in buffers))
        self.assertTrue(all(not bool(torch.any(buffer.cpu())) for buffer in buffers))

    def test_roots_match_independent_digest_and_full_native_tree(self):
        rng = np.random.default_rng(211)
        stream = torch.cuda.Stream(device='cuda:0')
        # Includes one leaf, nonzero tail offsets, odd leaf promotions, and a WR matrix.
        with torch.cuda.stream(stream):
            for key in (bytes(range(32)), bytes(reversed(range(32)))):
                for size in (16, 17, 128, 1024, 1040, 2048, 3072, 5120, 129*2048):
                    with self.subTest(key=key[0], size=size):
                        with self.capture_words() as saved:
                            tree = native_cuda.ConstantTailTree(size, 0xE7, key, 'cuda:0')
                        self.assert_wiped(saved, 1)
                        for _ in range(3):
                            tail = rng.bytes(16)
                            raw = bytes([0xE7]) * (size - 16) + tail
                            expected = StreamDigest(raw + bytes(-size % 1024), key=key).digest()
                            signal = torch.from_numpy(np.frombuffer(raw, dtype=np.int8).copy()).to('cuda:0').view(1, -1)
                            with self.capture_words() as saved:
                                actual = tree.root(tail)
                            self.assert_wiped(saved, 2)
                            self.assertEqual(actual, expected)
                            self.assertEqual(actual, native_cuda.tree_root(signal, key))
                            self.assertEqual(key, tree.key)

    def test_build_and_update_wipe_keys_after_kernel_errors(self):
        key = bytes(range(32))
        with self.capture_words() as saved:
            with patch.object(native_cuda, '_check', side_effect=RuntimeError('build failure')):
                with self.assertRaisesRegex(RuntimeError, 'build failure'):
                    native_cuda.ConstantTailTree(3072, 64, key, 'cuda:0')
        self.assert_wiped(saved, 1)
        tree = native_cuda.ConstantTailTree(3072, 64, key, 'cuda:0')
        with self.capture_words() as saved:
            with patch.object(native_cuda, '_check', side_effect=RuntimeError('update failure')):
                with self.assertRaisesRegex(RuntimeError, 'update failure'):
                    tree.root(bytes(range(16)))
        self.assert_wiped(saved, 2)

    def test_partial_tail_allocation_failure_still_wipes_key(self):
        tree = native_cuda.ConstantTailTree(3072, 64, bytes(32), 'cuda:0')
        allocate = native_cuda._words
        saved = []
        def fail_second(value, device):
            if saved:
                raise MemoryError('tail staging failure')
            words = allocate(value, device)
            saved.append(words)
            return words
        with patch.object(native_cuda, '_words', side_effect=fail_second):
            with self.assertRaisesRegex(MemoryError, 'tail staging failure'):
                tree.root(bytes(16))
        self.assert_wiped(saved, 1)

    def test_tail_requires_sixteen_bytes(self):
        tree = native_cuda.ConstantTailTree(3072, 64, bytes(32), 'cuda:0')
        for tail in (bytes(15), bytes(17), bytearray(16)):
            with self.subTest(tail=type(tail)), self.assertRaises(ValueError):
                tree.root(tail)


if __name__ == '__main__':
    unittest.main()
