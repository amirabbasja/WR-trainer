import struct
import blake3
import numpy as np
import torch
from miner_utils import get_logger
from LRAEP_gateway.comm.dataclasses import CommitmentHash, OpenedBlockInfo
from LRAEP_gateway.comm.mining_configuration import PeriodicPattern
from miner_base.matmul_config import MatmulConfig
from .inner_hash import InnerHasher
_LOGGER = get_logger(__name__)
TRANSCRIPT_SIZE_U32 = 16
POW_TARGET_HARDEST = 0
POW_TARGET_EASIEST = 2 ** 256 - 1
HASH_ACCUMULATE_ROTATION = 13

def rotl32(x: np.uint32, n: int) -> np.uint32:
    return x << n | x >> 32 - n

class Transcript:

    def __init__(self) -> None:
        self.data: list[np.uint32] = [np.uint32(0)] * TRANSCRIPT_SIZE_U32

    def rotl_xor_into(self, reduction_count: int, combined_hash: np.uint32) -> None:
        idx = reduction_count % TRANSCRIPT_SIZE_U32
        self.data[idx] = rotl32(self.data[idx], HASH_ACCUMULATE_ROTATION) ^ combined_hash

def _num_hash_tiles_in_partial_tile(pattern: PeriodicPattern, remaining: int) -> int:
    max_elem = max(pattern.to_list())
    max_valid_offset = remaining - max_elem - 1
    if max_valid_offset < 0:
        return 0
    return sum((1 for off in range(max_valid_offset + 1) if pattern.offset_is_valid(off)))

def inner_hash_per_matmul(matmul_config: MatmulConfig, m: int, n: int, k: int) -> int:
    total = 0
    k_contribution = k // matmul_config.noise_rank * matmul_config.noise_rank
    matmul_tile_contribution = matmul_config.matmul_tile_h * matmul_config.matmul_tile_w * k_contribution
    num_full_matmul_tiles_h = m // matmul_config.matmul_tile_h
    num_full_matmul_tiles_w = n // matmul_config.matmul_tile_w
    num_full_matmul_tiles = num_full_matmul_tiles_h * num_full_matmul_tiles_w
    total += num_full_matmul_tiles * matmul_tile_contribution
    num_full_hash_tiles_h = num_full_matmul_tiles_h * (matmul_config.matmul_tile_h // matmul_config.hash_tile_h)
    num_full_hash_tiles_w = num_full_matmul_tiles_w * (matmul_config.matmul_tile_w // matmul_config.hash_tile_w)
    remaining_h = m % matmul_config.matmul_tile_h
    remaining_w = n % matmul_config.matmul_tile_w
    num_partial_hash_tiles_h = _num_hash_tiles_in_partial_tile(matmul_config.mining_config.rows_pattern, remaining_h)
    num_partial_hash_tiles_w = _num_hash_tiles_in_partial_tile(matmul_config.mining_config.cols_pattern, remaining_w)
    total_partial_hash_tiles = num_full_hash_tiles_h * num_partial_hash_tiles_w + num_full_hash_tiles_w * num_partial_hash_tiles_h + num_partial_hash_tiles_h * num_partial_hash_tiles_w
    hash_tile_contribution = matmul_config.hash_tile_h * matmul_config.hash_tile_w * k_contribution
    total += total_partial_hash_tiles * hash_tile_contribution
    return total

class NoisyGemm:

    def __init__(self, noise_range: int=128, noise_rank: int=128, hash_tile_h: int=16, hash_tile_w: int=16, matmul_tile_h: int=128, matmul_tile_w: int=128):
        if noise_range > 128:
            raise ValueError('noise_range must fit in uint7')
        if matmul_tile_h < hash_tile_h or matmul_tile_w < hash_tile_w:
            raise ValueError(f'matmul_tile_h={matmul_tile_h!r} and matmul_tile_w={matmul_tile_w!r} must be greater than or equal to hash_tile_h={hash_tile_h!r} and hash_tile_w={hash_tile_w!r}')
        self.noise_range = noise_range
        self.noise_rank = noise_rank
        self.hash_tile_h = hash_tile_h
        self.hash_tile_w = hash_tile_w
        self.matmul_tile_h = matmul_tile_h
        self.matmul_tile_w = matmul_tile_w
        self.inner_hash = InnerHasher(tile_h=hash_tile_h, tile_w=hash_tile_w)
        self.opened_block_info: OpenedBlockInfo | None = None
        self.num_inner_hashes = 0
        self.num_good_inner_hashes = 0
        _LOGGER.info(f'Initialized NoisyGemm with noise_range={noise_range!r}, noise_rank={noise_rank!r}, hash_tile_h={hash_tile_h!r}, hash_tile_w={hash_tile_w!r}, matmul_tile_h={matmul_tile_h!r}, matmul_tile_w={matmul_tile_w!r}')

    def __validate_matrix_range(self, tensor: torch.Tensor) -> None:
        data_range = 256 - self.noise_range
        min_data_value = -data_range // 2
        max_data_value = min_data_value + data_range - 1
        if torch.any(tensor < min_data_value) or torch.any(tensor > max_data_value):
            raise ValueError(f'tensor must be in the range [{min_data_value}, {max_data_value}], since noise range is {self.noise_range}')

    def __validate_noise_range(self, error_matrix: torch.Tensor, matrix: torch.Tensor) -> None:
        assert error_matrix.shape == matrix.shape, f'error_matrix.shape={error_matrix.shape!r}, matrix.shape={matrix.shape!r}'
        zero_point_translation = self.noise_range // 2
        min_noise_value = -zero_point_translation
        max_noise_value = self.noise_range - zero_point_translation
        assert torch.all(error_matrix >= min_noise_value) and torch.all(error_matrix <= max_noise_value), f'noise must be in the range [{min_noise_value}, {max_noise_value}]'

    def noise_A(self, A: torch.Tensor, E_AL: torch.Tensor, E_AR: torch.Tensor, E_BL: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if A.dtype != torch.int8 or E_AL.dtype != torch.int8 or E_AR.dtype != torch.int8 or (E_BL.dtype != torch.int8):
            raise ValueError('A, E_AL, E_AR, E_BL must be int8')
        self.__validate_matrix_range(A)
        if A.shape[0] != E_AL.shape[0]:
            raise ValueError(f'A.shape[0]={A.shape[0]!r}, E_AL.shape[0]={E_AL.shape[0]!r}, expected shapes are A: m x k, E_AL: m x r')
        if not A.shape[1] == E_AR.shape[1] == E_BL.shape[0]:
            raise ValueError(f'A.shape[1]={A.shape[1]!r}, E_AR.shape[1]={E_AR.shape[1]!r}, E_BL.shape[0]={E_BL.shape[0]!r}, expected shapes are A: m x k, E_AR: r x k, E_BL: k x r')
        if not E_AL.shape[1] == E_AR.shape[0] == E_BL.shape[1]:
            raise ValueError(f'E_AL.shape[1]={E_AL.shape[1]!r}, E_AR.shape[0]={E_AR.shape[0]!r}, E_BL.shape[1]={E_BL.shape[1]!r}, expected shapes are E_AL: m x r, E_AR: r x k, E_BL: k x r')
        E_A = torch.matmul(E_AL.to(torch.int32), E_AR.to(torch.int32))
        self.__validate_noise_range(E_A, A)
        E_A = E_A.to(torch.int8)
        A_noised = A + E_A
        A_E_BL = torch.matmul(A.to(torch.int32), E_BL.to(torch.int32))
        return (A_noised, A_E_BL)

    def noise_B(self, B: torch.Tensor, E_AR: torch.Tensor, E_BL: torch.Tensor, E_BR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if B.dtype != torch.int8 or E_AR.dtype != torch.int8 or E_BL.dtype != torch.int8 or (E_BR.dtype != torch.int8):
            raise ValueError('B, E_AR, E_BL, E_BR must be int8')
        self.__validate_matrix_range(B)
        if B.shape[1] != E_BR.shape[1]:
            raise ValueError(f'B.shape[1]={B.shape[1]!r}, E_BR.shape[1]={E_BR.shape[1]!r}, expected shapes are B: k x n, E_BR: r x n')
        if not B.shape[0] == E_AR.shape[1] == E_BL.shape[0]:
            raise ValueError(f'B.shape[0]={B.shape[0]!r}, E_AR.shape[1]={E_AR.shape[1]!r}, E_BL.shape[0]={E_BL.shape[0]!r}, expected shapes are B: k x n, E_AR: r x k, E_BL: k x r')
        if not E_AR.shape[0] == E_BL.shape[1] == E_BR.shape[0]:
            raise ValueError(f'E_AR.shape[0]={E_AR.shape[0]!r}, E_BL.shape[1]={E_BL.shape[1]!r}, E_BR.shape[0]={E_BR.shape[0]!r}, expected shapes are E_AR: r x k, E_BL: k x r, E_BR: r x n')
        E_B = torch.matmul(E_BL.to(torch.int32), E_BR.to(torch.int32))
        self.__validate_noise_range(E_B, B)
        E_B = E_B.to(torch.int8)
        B_noised = B + E_B
        EAR_BpEB = torch.matmul(E_AR.to(torch.int32), B_noised.to(torch.int32))
        return (B_noised, EAR_BpEB)

    def _accumulate_transcripts(self, transcripts: list[list[Transcript]], reduction_count: int, C_block: torch.Tensor) -> None:
        hash_results = self.inner_hash.hash_tensor(C_block)
        self.num_inner_hashes += len(hash_results)
        for hash_result in hash_results:
            hi, wi = hash_result.index
            transcript = transcripts[hi][wi]
            transcript.rotl_xor_into(reduction_count, hash_result.hash)

    def _check_pow_target(self, transcript: Transcript, pow_key: bytes, pow_target: int) -> bool:
        transcript_bytes = b''.join((struct.pack('<I', int(w)) for w in transcript.data))
        hash_result = blake3.blake3(transcript_bytes, key=pow_key).digest()
        hash_int = int.from_bytes(hash_result, 'little')
        return hash_int <= pow_target

    def _record_opened_block(self, m: int, n: int) -> None:
        self.num_good_inner_hashes += 1
        _LOGGER.debug(f'Found valid block at m={m!r}, n={n!r}')
        self.opened_block_info = OpenedBlockInfo(A_row_indices=list(range(m, m + self.hash_tile_h)), B_column_indices=list(range(n, n + self.hash_tile_w)), A=None, B_t=None, commitment_hash=None, noise_rank=self.noise_rank)
        _LOGGER.debug(f'A_row_indices={self.opened_block_info.A_row_indices}, B_column_indices={self.opened_block_info.B_column_indices}, ')

    def _process_output_tile(self, A: torch.Tensor, B: torch.Tensor, k: int, i: int, i_max: int, j: int, j_max: int) -> tuple[torch.Tensor, bool, list[list[Transcript]]]:
        block_h = i_max - i
        block_w = j_max - j
        hash_tile_h = self.inner_hash.tile_h
        hash_tile_w = self.inner_hash.tile_w
        num_hash_tiles_h = block_h // hash_tile_h
        num_hash_tiles_w = block_w // hash_tile_w
        transcripts: list[list[Transcript]] = [[Transcript() for _ in range(num_hash_tiles_w)] for _ in range(num_hash_tiles_h)]
        reduction_count = 0
        has_full_tiles = False
        C_block = torch.zeros((block_h, block_w), dtype=torch.int32)
        for p in range(0, k, self.noise_rank):
            p_max = min(p + self.noise_rank, k)
            A_tile = A[i:i_max, p:p_max]
            B_tile = B[p:p_max, j:j_max]
            C_tile = torch.matmul(A_tile.to(torch.int32), B_tile.to(torch.int32))
            C_block += C_tile
            is_full_tile = block_h >= hash_tile_h and block_w >= hash_tile_w and (p_max - p == self.noise_rank)
            if is_full_tile:
                self._accumulate_transcripts(transcripts=transcripts, reduction_count=reduction_count, C_block=C_block)
                reduction_count += 1
                has_full_tiles = True
        return (C_block, has_full_tiles, transcripts)

    def _check_tile_transcripts(self, transcripts: list[list[Transcript]], i: int, j: int, pow_key: bytes, pow_target: int) -> bool:
        hash_tile_h = self.inner_hash.tile_h
        hash_tile_w = self.inner_hash.tile_w
        num_hash_tiles_h = len(transcripts)
        num_hash_tiles_w = len(transcripts[0]) if num_hash_tiles_h > 0 else 0
        for hi in range(num_hash_tiles_h):
            for wi in range(num_hash_tiles_w):
                if self._check_pow_target(transcripts[hi][wi], pow_key, pow_target):
                    self._record_opened_block(m=i + hi * hash_tile_h, n=j + wi * hash_tile_w)
                    return True
        return False

    def _tiled_matmul(self, A: torch.Tensor, B: torch.Tensor, pow_key: bytes, pow_target: int) -> tuple[torch.Tensor, bool]:
        assert A.shape[1] == B.shape[0]
        m, k = A.shape
        n = B.shape[1]
        C = torch.zeros((m, n), dtype=torch.int32)
        hash_tile_h = self.inner_hash.tile_h
        hash_tile_w = self.inner_hash.tile_w
        if m < hash_tile_h or n < hash_tile_w or self.noise_rank < min(hash_tile_h, hash_tile_w):
            raise ValueError(f'm={m!r}, n={n!r}, hash_tile_h={hash_tile_h!r}, hash_tile_w={hash_tile_w!r}, self.noise_rank={self.noise_rank!r}, matmul tile should be larger than hash tile and noise rank')
        found_block = False
        for i in range(0, m, self.noise_rank):
            i_max = min(i + self.noise_rank, m)
            for j in range(0, n, self.noise_rank):
                j_max = min(j + self.noise_rank, n)
                C_block, has_full_tiles, transcripts = self._process_output_tile(A, B, k, i, i_max, j, j_max)
                if not found_block and has_full_tiles:
                    found_block = self._check_tile_transcripts(transcripts, i, j, pow_key, pow_target)
                C[i:i_max, j:j_max] = C_block
        return (C, found_block)

    def get_opened_block_info(self) -> OpenedBlockInfo | None:
        return self.opened_block_info

    def gemm(self, A_noised: torch.Tensor, B_noised: torch.Tensor, E_AL: torch.Tensor, E_BR: torch.Tensor, A_E_BL: torch.Tensor, EAR_BpEB: torch.Tensor, pow_key: bytes, pow_target: int) -> tuple[torch.Tensor, bool]:
        if A_noised.dtype != torch.int8 or B_noised.dtype != torch.int8 or E_AL.dtype != torch.int8 or (E_BR.dtype != torch.int8) or (A_E_BL.dtype != torch.int32) or (EAR_BpEB.dtype != torch.int32):
            raise ValueError('Invalid dtypes: A_noised, B_noised, E_AL, E_BR, must be int8; A_E_BL, EAR_BpEB must be int32')
        if A_noised.shape[1] != B_noised.shape[0]:
            raise ValueError(f'A_noised.shape[1]={A_noised.shape[1]!r}, B_noised.shape[0]={B_noised.shape[0]!r}, expected shapes are A_noised: m x k, B_noised: k x n')
        if not A_noised.shape[0] == E_AL.shape[0] == A_E_BL.shape[0]:
            raise ValueError(f'A_noised.shape[0]={A_noised.shape[0]!r}, E_AL.shape[0]={E_AL.shape[0]!r}, A_E_BL.shape[0]={A_E_BL.shape[0]!r}, expected shapes are A_noised: m x k, E_AL: m x r, A_E_BL: m x r')
        if not B_noised.shape[1] == E_BR.shape[1] == EAR_BpEB.shape[1]:
            raise ValueError(f'B_noised.shape[1]={B_noised.shape[1]!r}, E_BR.shape[1]={E_BR.shape[1]!r}, EAR_BpEB.shape[1]={EAR_BpEB.shape[1]!r}, expected shapes are B_noised: k x n, E_BR: r x n, EAR_BpEB: r x n')
        if not E_AL.shape[1] == E_BR.shape[0] == A_E_BL.shape[1] == EAR_BpEB.shape[0]:
            raise ValueError(f'E_AL.shape[1]={E_AL.shape[1]!r}, E_BR.shape[0]={E_BR.shape[0]!r}, A_E_BL.shape[1]={A_E_BL.shape[1]!r}, EAR_BpEB.shape[0]={EAR_BpEB.shape[0]!r}, expected shapes are E_AL: m x r, E_BR: r x n, A_E_BL: m x r, EAR_BpEB: r x n')
        A_EB = torch.matmul(A_E_BL, E_BR.to(torch.int32))
        EA_BpEB = torch.matmul(E_AL.to(torch.int32), EAR_BpEB)
        C_noised, found_block = self._tiled_matmul(A_noised, B_noised, pow_key, pow_target)
        C = C_noised - A_EB - EA_BpEB
        return (C, found_block)

    def noisy_gemm(self, A: torch.Tensor, B: torch.Tensor, E_AL: torch.Tensor, E_AR: torch.Tensor, E_BL: torch.Tensor, E_BR: torch.Tensor, commitment_hash: CommitmentHash, pow_target: int=POW_TARGET_HARDEST) -> tuple[torch.Tensor, bool]:
        if A.dtype != torch.int8 or B.dtype != torch.int8 or E_AL.dtype != torch.int8 or (E_AR.dtype != torch.int8) or (E_BL.dtype != torch.int8) or (E_BR.dtype != torch.int8):
            raise ValueError('A, B, E_AL, E_AR, E_BL, E_BR must be int8')
        self.__validate_matrix_range(A)
        self.__validate_matrix_range(B)
        if A.shape[0] != E_AL.shape[0]:
            raise ValueError(f'A.shape[0]={A.shape[0]!r}, E_AL.shape[0]={E_AL.shape[0]!r}, expected shapes are A: m x k, E_AL: m x r')
        if not A.shape[1] == B.shape[0] == E_AR.shape[1] == E_BL.shape[0]:
            raise ValueError(f'A.shape[1]={A.shape[1]!r}, B.shape[0]={B.shape[0]!r}, E_AR.shape[1]={E_AR.shape[1]!r}, E_BL.shape[0]={E_BL.shape[0]!r}, expected shapes are A: m x k, B: k x n, E_AR: r x k, E_BL: k x r')
        if B.shape[1] != E_BR.shape[1]:
            raise ValueError(f'B.shape[1]={B.shape[1]!r}, E_BR.shape[1]={E_BR.shape[1]!r}, expected shapes are B: k x n, E_BR: r x n')
        if not E_AL.shape[1] == E_AR.shape[0] == E_BL.shape[1] == E_BR.shape[0]:
            raise ValueError(f'E_AL.shape[1]={E_AL.shape[1]!r}, E_AR.shape[0]={E_AR.shape[0]!r}, E_BL.shape[1]={E_BL.shape[1]!r}, E_BR.shape[0]={E_BR.shape[0]!r}, expected shapes are E_AL: m x r, E_AR: r x k, E_BL: k x r, E_BR: r x n')
        if A.shape[0] < self.matmul_tile_h or A.shape[1] < self.noise_rank or B.shape[1] < self.matmul_tile_w:
            raise ValueError(f'A.shape[0]={A.shape[0]!r}, A.shape[1]={A.shape[1]!r}, B.shape[1]={B.shape[1]!r}, self.matmul_tile_h={self.matmul_tile_h!r}, self.noise_rank={self.noise_rank!r}, self.matmul_tile_w={self.matmul_tile_w!r}, expected A.shape[0] and A.shape[1] and B.shape[1] to be greater than matmul_tile_h and noise_rank and matmul_tile_w')
        if not POW_TARGET_HARDEST <= pow_target <= POW_TARGET_EASIEST:
            raise ValueError(f'pow_target must be in the range pow_target={pow_target:#x} [{POW_TARGET_HARDEST:#x}, {POW_TARGET_EASIEST:#x}]')
        A_noised, A_E_BL = self.noise_A(A, E_AL, E_AR, E_BL)
        B_noised, EAR_BpEB = self.noise_B(B, E_AR, E_BL, E_BR)
        result, found_block = self.gemm(A_noised, B_noised, E_AL, E_BR, A_E_BL, EAR_BpEB, commitment_hash.noise_seed_A, pow_target)
        if found_block:
            assert self.opened_block_info.A is None
            assert self.opened_block_info.B_t is None
            assert self.opened_block_info.commitment_hash is None
            self.opened_block_info.commitment_hash = commitment_hash
            self.opened_block_info.A = A
            self.opened_block_info.B_t = B.T.contiguous()
        return (result, found_block)
