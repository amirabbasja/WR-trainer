from functools import lru_cache
import numpy as np
import torch
_SEED_A = b'A_tensor' + bytes(24)
_SEED_B = b'B_tensor' + bytes(24)

def _permutation_indices(generator, seed, key, count):
    assert len(seed) == 32, 'seed must be 32 bytes'
    assert len(key) == 32, 'key must be 32 bytes'
    random_bytes = b''.join((generator._NoiseGenerator__get_random_hash(index, seed, key, 1) for index in range((count + 7) // 8)))
    values = np.frombuffer(random_bytes, dtype='<u4')[:count]
    first = (values & generator.rank_mask).astype(np.int64)
    high = values.astype(np.uint64) * np.uint64(generator.noise_rank - 1) >> np.uint64(32)
    second = first ^ (high + np.uint64(1)).astype(np.int64)
    return (first, second)

@lru_cache(maxsize=1)
def _generator():
    from miner_base.noise_generation import NoiseGenerator

    class VectorizedNoiseGenerator(NoiseGenerator):

        def _NoiseGenerator__generate_permutation_matrix(self, seed, key, rows, cols, assign_columns):
            assert rows == self.noise_rank or cols == self.noise_rank
            assert len(seed) == 32, 'seed must be 32 bytes'
            assert len(key) == 32, 'key must be 32 bytes'
            if assign_columns:
                assert rows == self.noise_rank
                count = cols
            else:
                assert cols == self.noise_rank
                count = rows
            first, second = _permutation_indices(self, seed, key, count)
            matrix = np.zeros((count, self.noise_rank), dtype=np.int8)
            lines = np.arange(count)
            matrix[lines, first] = 1
            matrix[lines, second] = -1
            if assign_columns:
                matrix = np.ascontiguousarray(matrix.T)
            return torch.from_numpy(matrix)
    return VectorizedNoiseGenerator(noise_rank=128)

def generate_noise_factors(seed_a: bytes, seed_b: bytes, m: int, k: int, n: int):
    return _generator().generate_noise_metrices(seed_a, seed_b, m, k, n)

def generate_sparse_noise_factors(seed_a: bytes, seed_b: bytes, m: int, k: int, n: int):
    generator = _generator()
    rank = generator.noise_rank
    if m < rank:
        raise ValueError(f'A_rows must be greater than or equal to noise_rank, got {m}')
    if k < rank:
        raise ValueError(f'common_dim must be greater than or equal to noise_rank, got {k}')
    if n < rank:
        raise ValueError(f'B_cols must be greater than or equal to noise_rank, got {n}')
    uniform = generator._NoiseGenerator__generate_uniform_random_matrix
    al = uniform(_SEED_A, seed_a, m)
    a_first, a_second = _permutation_indices(generator, _SEED_A, seed_a, k)
    b_first, b_second = _permutation_indices(generator, _SEED_B, seed_b, k)
    br = uniform(_SEED_B, seed_b, n).T
    return (al, torch.from_numpy(a_first), torch.from_numpy(a_second), torch.from_numpy(b_first), torch.from_numpy(b_second), br)
