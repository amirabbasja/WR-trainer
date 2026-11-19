"""WR constant-operand cache lifetime and equivalence checks."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from workrunner import native_cuda
from workrunner.protocol import TrainingConfig, load_official_helpers
from workrunner.stream_digest import StreamDigest
from workrunner.training_protocol import TrainingJob
from workrunner.training_work import CapturedBatchPreparation, prepare_training_batch


ROOT = Path(__file__).resolve().parents[1]


def make_job(*, header_tag=0, device_seed=bytes(range(8))):
    header = bytes([header_tag]) + bytes(71) + (0x207fffff).to_bytes(4, 'little')
    return TrainingJob('offline-cache-test', header, 0, 3, 1, update_payload=bytes(200), device_seed=device_seed)


class PreparationCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_official_helpers(ROOT / 'src' / 'work_helpers')

    def setUp(self):
        self.cfg = TrainingConfig(m=128, n=128, k=2048)
        self.preparation = CapturedBatchPreparation(self.cfg, 'cpu')
        self.rng = np.random.default_rng(73)
        self.noise_calls = []

        self.tail_roots = []

        def tail_tree(size, value, key, device):
            def root(tail):
                return StreamDigest(bytes([value]) * (size - 16) + tail + bytes(-size % 1024), key=key).digest()
            tree = Mock()
            tree.root.side_effect = root
            self.tail_roots.append(tree.root)
            return tree

        def constant_root(size, value, key, device):
            # Only the right input root is constant; its header binding still matters.
            return StreamDigest(bytes([value]) * size, key=key).digest()

        def noise(rows, k, value, key, device, *, right=False):
            self.noise_calls.append((key, right))
            return torch.full((rows, k), key[0] % 128 - 64, dtype=torch.int8, device=device)

        self.tail_tree = patch.object(native_cuda, 'ConstantTailTree', side_effect=tail_tree).start()
        self.constant_root = patch.object(native_cuda, 'constant_tree_root', side_effect=constant_root).start()
        self.noise = patch.object(native_cuda, 'constant_noised_operand', side_effect=noise).start()
        self.addCleanup(patch.stopall)

    def prepare(self, job):
        return prepare_training_batch(job, self.cfg, self.rng, device='cpu', native=True,
                                      preparation=self.preparation)

    def test_new_nonce_reuses_only_the_right_operand(self):
        job = make_job()
        first, second = self.prepare(job), self.prepare(job)
        self.assertIs(first.b_noised_t, second.b_noised_t)
        self.assertIsNot(first.a_noised, second.a_noised)
        self.assertNotEqual(first.local_seed, second.local_seed)
        self.assertNotEqual(first.commitment.noise_seed_A, second.commitment.noise_seed_A)
        self.assertEqual(first.commitment.noise_seed_B, second.commitment.noise_seed_B)
        self.assertEqual(self.constant_root.call_count, 1)
        self.assertEqual(self.tail_tree.call_count, 1)
        self.assertEqual(self.tail_roots[0].call_count, 2)
        self.assertEqual([right for key, right in self.noise_calls], [True, False, False])
        self.assertTrue(first.native)
        self.assertEqual(first.a_noised.shape, (self.cfg.m, self.cfg.k))

    def test_header_change_rebuilds_right_root_and_noise(self):
        first = self.prepare(make_job())
        second = self.prepare(make_job(header_tag=1))
        self.assertIsNot(first.b_noised_t, second.b_noised_t)
        self.assertNotEqual(first.commitment.noise_seed_B, second.commitment.noise_seed_B)
        self.assertEqual(self.constant_root.call_count, 2)
        self.assertEqual(self.tail_tree.call_count, 2)
        self.assertEqual([right for key, right in self.noise_calls], [True, False, True, False])

    def test_server_seed_change_preserves_right_cache_and_updates_left_seed(self):
        first = self.prepare(make_job())
        second = self.prepare(make_job(device_seed=bytes(reversed(range(8)))))
        self.assertIs(first.b_noised_t, second.b_noised_t)
        self.assertNotEqual(first.commitment.noise_seed_A, second.commitment.noise_seed_A)
        self.assertEqual(self.constant_root.call_count, 1)

    def test_clear_releases_cache_and_next_batch_rebuilds_it(self):
        first = self.prepare(make_job())
        self.preparation.clear()
        self.assertIsNone(self.preparation._right_operand)
        self.assertIsNone(self.preparation._left_tree)
        second = self.prepare(make_job())
        self.assertIsNot(first.b_noised_t, second.b_noised_t)
        self.assertEqual(self.constant_root.call_count, 2)
        self.assertEqual(self.tail_tree.call_count, 2)

    def test_rejects_mismatched_configuration_and_reference_mode(self):
        for cfg, native in ((replace(self.cfg, k=4096), True), (self.cfg, False)):
            with self.subTest(cfg=cfg, native=native), self.assertRaises(ValueError):
                prepare_training_batch(make_job(), cfg, self.rng, device='cpu', native=native,
                                       preparation=self.preparation)
        self.assertEqual(self.tail_tree.call_count, 0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
class NativePreparationEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_official_helpers(ROOT / 'src' / 'work_helpers')
        native_cuda.load_native('cuda:0')

    def test_contiguous_cache_matches_uncached_operands_commitments_and_transcripts(self):
        cfg = TrainingConfig(m=128, n=128, k=2048)
        device = torch.device('cuda:0')
        preparation = CapturedBatchPreparation(cfg, device)
        # Include cache reuse and a new header, each with a fresh reserved seed row.
        for index, job in enumerate((make_job(), make_job(), make_job(header_tag=1))):
            with self.subTest(index=index):
                seed = 110 + index
                plain = prepare_training_batch(job, cfg, np.random.default_rng(seed), device=device, native=True)
                cached = prepare_training_batch(job, cfg, np.random.default_rng(seed), device=device,
                                                native=True, preparation=preparation)
                self.assertTrue(cached.native)
                self.assertEqual(cached.local_seed, plain.local_seed)
                self.assertEqual(cached.device_seed, plain.device_seed)
                self.assertEqual(cached.header_bytes, plain.header_bytes)
                self.assertEqual(cached.commitment, plain.commitment)
                self.assertTrue(torch.equal(cached.a_noised, plain.a_noised))
                self.assertTrue(torch.equal(cached.b_noised_t, plain.b_noised_t))
                expected = native_cuda.transcripts_device(plain.a_noised, plain.b_noised_t)
                actual = native_cuda.transcripts_device(cached.a_noised, cached.b_noised_t)
                self.assertTrue(torch.equal(actual, expected))
                import workrunner.protocol as protocol
                compute = getattr(protocol, 'compute_batch', None) or protocol.search_batch
                self.assertEqual(list(compute(cached, 1, device, lambda: False)), [])


if __name__ == '__main__':
    unittest.main()
