"""CUDA transcript checks against an independent NumPy implementation.

Run from a checkout with:
    PYTHONPATH=src python -m unittest discover -s tests -p test_native_transcripts.py
The CPU reference deliberately accumulates full dot products across rank-128
slabs before reducing each 8x8 or 8x16 tile, matching the attested wire transcript.
"""
import unittest

import numpy as np
import torch

from workrunner import native_cuda
from workrunner.stream_digest import StreamDigest


def reference_transcripts(a, bt, tile_w=16):
    """Return uint32 transcripts without calling the CUDA/helper implementation."""
    m, k = a.shape
    n = bt.shape[0]
    cells = np.zeros((m, n), dtype=np.int32)
    words = np.zeros((m // 8, n // tile_w, 16), dtype=np.uint32)
    for slab, start in enumerate(range(0, k, 128)):
        cells += a[:, start:start + 128].astype(np.int32) @ bt[:, start:start + 128].astype(np.int32).T
        tiles = cells.view(np.uint32).reshape(m // 8, 8, n // tile_w, tile_w)
        reduction = np.bitwise_xor.reduce(np.bitwise_xor.reduce(tiles, axis=3), axis=1)
        slot = slab % 16
        previous = words[:, :, slot]
        words[:, :, slot] = ((previous << np.uint32(13)) | (previous >> np.uint32(19))) ^ reduction
    return words


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class NativeTranscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compilation failures should fail on a CUDA host, not silently skip.
        native_cuda.load_native('cuda:0')

    def arrays(self, m, n, k, seed=453):
        rng = np.random.default_rng(seed)
        return (rng.integers(-128, 128, size=(m, k), dtype=np.int8),
                rng.integers(-128, 128, size=(n, k), dtype=np.int8))

    def check_case(self, m, n, k, *, offset=0, extreme=False, tile_w=16, variant=0):
        a, bt = self.arrays(m, n, k)
        if extreme:
            # Exercise signed packing and long-lived accumulators with large,
            # nonzero cell values as well as the complete int8 random domain.
            a[:m // 2] = -128
            bt[:n // 2] = 127
        expected = reference_transcripts(a, bt, tile_w=tile_w)
        if offset:
            a_storage = torch.empty(a.size + offset, dtype=torch.int8, device='cuda:0')
            b_storage = torch.empty(bt.size + offset, dtype=torch.int8, device='cuda:0')
            ag = a_storage[offset:].view(m, k)
            bg = b_storage[offset:].view(n, k)
            ag.copy_(torch.from_numpy(a))
            bg.copy_(torch.from_numpy(bt))
            self.assertEqual(ag.data_ptr() % 16, offset)
            self.assertEqual(bg.data_ptr() % 16, offset)
        else:
            ag, bg = torch.from_numpy(a).cuda(), torch.from_numpy(bt).cuda()
        actual = native_cuda.transcripts_device(ag, bg, tile_w=tile_w, variant=variant)
        np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32), expected)
        return actual, expected

    def test_cumulative_transcripts_full_int8_and_partial_blocks(self):
        for tile_w in (8, 16):
            for m, n, k in ((8, 16, 128), (24, 48, 2048), (72, 80, 8192), (128, 144, 16384)):
                with self.subTest(tile_w=tile_w, m=m, n=n, k=k):
                    self.check_case(m, n, k, tile_w=tile_w, extreme=k >= 8192)

    def test_all_existing_8x8_variants_match_reference_at_block_edges(self):
        for variant in range(1, 8):
            with self.subTest(variant=variant):
                self.check_case(72, 80, 2048, tile_w=8, variant=variant)

    def test_large_8x8_default_dispatch_matches_reference(self):
        self.check_case(776, 784, 128, tile_w=8)

    def test_large_optimized_dispatch_and_long_transcripts(self):
        # This enters the large-matrix dispatch and updates each transcript slot
        # repeatedly. Compare every 8x8 word with the retained scalar kernel,
        # and independently reconstruct corner tiles for both transcript widths.
        m, n, k = 776, 784, 16384
        a, bt = self.arrays(m, n, k, seed=979)
        ag, bg = torch.from_numpy(a).cuda(), torch.from_numpy(bt).cuda()
        for tile_w in (8, 16):
            with self.subTest(tile_w=tile_w):
                actual = native_cuda.transcripts_device(ag, bg, tile_w=tile_w)
                actual_cpu = actual.cpu().numpy().view(np.uint32)
                if tile_w == 8:
                    legacy = native_cuda.transcripts_device(ag, bg, tile_w=8, variant=7)
                    np.testing.assert_array_equal(actual_cpu, legacy.cpu().numpy().view(np.uint32))
                for row in (0, m // 8 - 1):
                    for col in (0, n // tile_w - 1):
                        expected = reference_transcripts(a[row * 8:(row + 1) * 8],
                                                         bt[col * tile_w:(col + 1) * tile_w],
                                                         tile_w=tile_w)
                        np.testing.assert_array_equal(actual_cpu[row, col], expected[0, 0])

    def test_nondefault_cuda_stream(self):
        a, bt = self.arrays(72, 80, 2048, seed=980)
        ag, bg = torch.from_numpy(a).cuda(), torch.from_numpy(bt).cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for tile_w in (8, 16):
            with self.subTest(tile_w=tile_w):
                with torch.cuda.stream(stream):
                    actual = native_cuda.transcripts_device(ag, bg, tile_w=tile_w)
                stream.synchronize()
                np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32),
                                              reference_transcripts(a, bt, tile_w=tile_w))

    def test_unaligned_vector_fallback_matches_reference(self):
        for tile_w in (8, 16):
            with self.subTest(tile_w=tile_w):
                self.check_case(24, 48, 2048, offset=4, tile_w=tile_w)

    def test_candidate_digests_capacity_retry_and_threshold_boundary(self):
        transcripts, expected = self.check_case(24, 48, 2048)
        key = bytes(range(32))
        candidates = []
        for row in range(expected.shape[0]):
            for col in range(expected.shape[1]):
                payload = expected[row, col].astype('<u4', copy=False).tobytes()
                digest = StreamDigest(payload, key=key).digest()
                candidates.append((row, col, digest))
        self.assertEqual(native_cuda.match_candidates(transcripts, key, 2**256 - 1, capacity=1), candidates)
        boundary = int.from_bytes(candidates[3][2], 'little')
        for threshold in (0, boundary - 1, boundary):
            with self.subTest(threshold=threshold):
                wanted = [item for item in candidates if int.from_bytes(item[2], 'little') <= threshold]
                self.assertEqual(native_cuda.match_candidates(transcripts, key, threshold, capacity=2), wanted)


if __name__ == '__main__':
    unittest.main()
