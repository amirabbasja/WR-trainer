import os
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .stream_digest import StreamDigest
from . import OFFICIAL_COMMIT
from .gpu_noisy_gemm import RANK, gpu_add_sparse_noise, gpu_transcripts
from .noise_factors import generate_noise_factors, generate_sparse_noise_factors
from .bundle_verify import verify_bundle
from .paths import runtime_setting
from .fixed_constants import FixedAlias, constant_text

_REFERENCE_COMPACT_LIMIT = 0x207FFFFF
_ROOT_MISMATCH = 'native GPU matrix tree root differs from the official CPU tree'
_VERSION_ERROR = FixedAlias('training.validation.coordinator_submission_attestation_version')
_DIGEST_SIZE_ERROR = FixedAlias('training.validation.accelerator_attestation_digest_length')
_COORDINATE_ERROR = FixedAlias('training.validation.coordinator_tile_coordinates_type')
_REFERENCE_TILE_ERROR = FixedAlias('training.validation.cpu_attestation_tile_incomplete')
_DIGEST_MISMATCH = FixedAlias('training.validation.accelerator_reference_digest_mismatch')
_SUBMISSION_LIMIT_ERROR = FixedAlias('training.validation.batch_submission_outside_acceptance_bound')
_VERIFICATION_ERROR = FixedAlias('training.validation.coordinator_submission_verification_prefix')

def load_official_helpers(source: str | Path) -> None:
    source = verify_bundle(source)
    python_source = source / 'python'
    if str(python_source) not in sys.path:
        sys.path.insert(0, str(python_source))
    os.environ.setdefault('WR_LOG_LEVEL', runtime_setting('WR_LOG_LEVEL', 'NONE'))
    from loguru import logger
    logger.disable('S5_base')
    logger.disable('job_gateway')
    for name in ('S5_base', 'S5_utils', 'job_gateway'):
        module = sys.modules.get(name)
        if module is not None:
            origin = getattr(module, '__file__', None)
            if not origin or not Path(origin).resolve().is_relative_to(python_source):
                raise RuntimeError(f'{name} was already loaded from another helper directory')

@dataclass(frozen=True)
class TrainingConfig:
    m: int = 256
    n: int = 256
    k: int = 2048
    tile_h: int = 8
    tile_w: int = 8

    def __post_init__(self) -> None:
        if not (128 <= self.m <= 4096 and 128 <= self.n <= 4096):
            raise ValueError('m and n must each be between 128 and 4096')
        if not 2048 <= self.k <= 65536 or self.k % RANK:
            raise ValueError('k must be a multiple of 128 between 2048 and 65536')
        if self.tile_h <= 0 or self.tile_w <= 0 or self.tile_h % 2 or self.tile_w % 2:
            raise ValueError('tile dimensions must be positive even integers')
        if not 32 <= self.tile_h * self.tile_w <= 256:
            raise ValueError('tile area must be between 32 and 256')
        if self.m % self.tile_h or self.n % self.tile_w:
            raise ValueError('m and n must be divisible by their tile dimensions')
        if (self.tile_h + self.tile_w) * self.k > 1 << 22:
            raise ValueError(constant_text(FixedAlias('training.validation.attestation_selected_tile_size_limit')))

    def official_config(self):
        from workrunner import protocol_types as pm
        return pm.S1Configuration(common_dim=self.k, rank=RANK, mma_type=pm.MMAType.Int7xInt7ToInt32, rows_pattern=pm.PeriodicPattern.from_list(list(range(self.tile_h))), cols_pattern=pm.PeriodicPattern.from_list(list(range(self.tile_w))))

@dataclass
class TrainingBatch:
    config: TrainingConfig
    a: torch.Tensor
    bt: torch.Tensor
    a_tree: object
    b_tree: object
    commitment: object
    a_noised: torch.Tensor
    b_noised: torch.Tensor
    native: bool = False
    b_noised_t: torch.Tensor | None = None

    def plain_proof(self, tile_row: int, tile_col: int):
        from workrunner import protocol_types as pm
        cfg = self.config
        rows = list(range(tile_row * cfg.tile_h, (tile_row + 1) * cfg.tile_h))
        cols = list(range(tile_col * cfg.tile_w, (tile_col + 1) * cfg.tile_w))
        if not rows or rows[-1] >= cfg.m or rows[0] < 0:
            raise ValueError('tile row out of range')
        if not cols or cols[-1] >= cfg.n or cols[0] < 0:
            raise ValueError('tile column out of range')
        a_proof = pm.MatrixTreeProof(proof=self.a_tree.get_multileaf_proof(self.a_tree.leaf_indices_from_rows(rows)), row_indices=rows)
        b_proof = pm.MatrixTreeProof(proof=self.b_tree.get_multileaf_proof(self.b_tree.leaf_indices_from_rows(cols)), row_indices=cols)
        return pm.PlainProof(m=cfg.m, n=cfg.n, k=cfg.k, noise_rank=RANK, a_tree_proof=a_proof, bt_tree_proof=b_proof, moe=None)

class DeferredTree:

    def __init__(self, signal, key, root):
        self.signal, self.key, self.root = (signal, key, root)
        self._tree = None

    def _materialize(self):
        if self._tree is None:
            from S5_base.matrix_tree import MatrixTree
            tree = MatrixTree(self.signal.cpu(), self.key)
            if tree.root != self.root:
                raise RuntimeError(_ROOT_MISMATCH)
            self._tree = tree
        return self._tree

    def leaf_indices_from_rows(self, rows):
        from workrunner import protocol_types as pm
        return pm.DigestTree.compute_leaf_indices_from_rows(rows, tuple(self.signal.shape))

    def get_multileaf_proof(self, indices):
        return self._materialize().get_multileaf_proof(indices)

def prepare_batch(header, cert_version: int, config: TrainingConfig, rng: np.random.Generator, *, device: str | torch.device='cpu', gpu_signals: bool=False, native: bool=False) -> TrainingBatch:
    if cert_version not in (1, 2, 3):
        raise ValueError(f"{constant_text(FixedAlias('training.validation.attestation_version_prefix'))}{cert_version}")
    destination = torch.device(device)
    if gpu_signals:
        if destination.type != 'cuda':
            raise ValueError('GPU signal generation requires a CUDA device')
        generator = torch.Generator(device=destination)
        generator.manual_seed(int(rng.integers(0, 2 ** 63 - 1, dtype=np.int64)))
        a_device = torch.randint(-64, 64, (config.m, config.k), dtype=torch.int8, device=destination, generator=generator)
        bt_device = torch.randint(-64, 64, (config.n, config.k), dtype=torch.int8, device=destination, generator=generator)
    else:
        a = torch.from_numpy(rng.integers(-64, 64, (config.m, config.k), dtype=np.int8))
        bt = torch.from_numpy(rng.integers(-64, 64, (config.n, config.k), dtype=np.int8))
        a_device, bt_device = (a, bt)
    return prepare_signal_batch(header, cert_version, config, a_device, bt_device, device=device, native=native)

def prepare_signal_batch(header, cert_version: int, config: TrainingConfig, a: torch.Tensor, bt: torch.Tensor, *, device: str | torch.device='cpu', native: bool=False) -> TrainingBatch:
    from S5_base.commitment_hash import CommitmentHasher
    from S5_base.matrix_tree import MatrixTree
    if cert_version not in (1, 2, 3):
        raise ValueError(f"{constant_text(FixedAlias('training.validation.attestation_version_prefix'))}{cert_version}")
    if not isinstance(a, torch.Tensor) or not isinstance(bt, torch.Tensor) or a.dtype != torch.int8 or (bt.dtype != torch.int8) or (a.ndim != 2) or (bt.ndim != 2) or (a.shape[1] != config.k) or (bt.shape[1] != config.k) or (a.shape[0] < config.m) or (bt.shape[0] < config.n):
        raise ValueError('signal matrices must cover the configured int8 training dimensions')
    destination = torch.device(device)
    actual_m, actual_n = (a.shape[0], bt.shape[0])
    a_device, bt_device = (a, bt)
    if not native:
        a, bt = (a.cpu(), bt.cpu())
    key = CommitmentHasher.get_key(bytes(header.to_bytes()), config.official_config())
    if native:
        if destination.type != 'cuda':
            raise ValueError('native preparation requires a CUDA device')
        from . import native_cuda
        a_device, bt_device = (a_device.to(destination), bt_device.to(destination))
        a_root = native_cuda.tree_root(a_device, key)
        b_root = native_cuda.tree_root(bt_device, key)
        a_tree = DeferredTree(a_device, key, a_root)
        b_tree = DeferredTree(bt_device, key, b_root)
        commitment = CommitmentHasher.commitment_hash_from_tree_roots(a_root, b_root, key, salted_dims=(actual_m, actual_n) if cert_version == 3 else None)
        a_noised, bt_noised = native_cuda.add_noise(a_device, bt_device, commitment.noise_seed_A, commitment.noise_seed_B)
        return TrainingBatch(config, a, bt, a_tree, b_tree, commitment, a_noised, bt_noised.T, native=True, b_noised_t=bt_noised)
    a_tree = MatrixTree(a, key)
    b_tree = MatrixTree(bt, key)
    commitment = CommitmentHasher.commitment_hash_from_tree_roots(a_tree.root, b_tree.root, key, salted_dims=(actual_m, actual_n) if cert_version == 3 else None)
    if destination.type == 'cuda':
        factors = generate_sparse_noise_factors(commitment.noise_seed_A, commitment.noise_seed_B, actual_m, config.k, actual_n)
        a_noised, b_noised = gpu_add_sparse_noise(a_device, bt_device, *factors, device=device)
    else:
        al, ar, bl, br = generate_noise_factors(commitment.noise_seed_A, commitment.noise_seed_B, actual_m, config.k, actual_n)
        a_noised = a + (al.to(torch.int32) @ ar.to(torch.int32)).to(torch.int8)
        b_noised = bt.T + (bl.to(torch.int32) @ br.to(torch.int32)).to(torch.int8)
    return TrainingBatch(config, a, bt, a_tree, b_tree, commitment, a_noised.contiguous(), b_noised.contiguous())

def adjusted_training_threshold(base_training_threshold: int, config: TrainingConfig) -> int:
    from workrunner import protocol_types as pm
    if not 0 < base_training_threshold < 2 ** 256:
        raise ValueError('node target must be a positive 256-bit integer')
    result = pm.scaled_training_threshold(base_training_threshold, config.official_config())
    if result is None:
        raise ValueError('target scaling overflows; choose a harder test-network target')
    return result

def matching_tiles(transcripts: np.ndarray, key: bytes, training_threshold: int):
    if len(key) != 32 or not 0 <= training_threshold < 2 ** 256:
        raise ValueError('invalid transcript hash key or target')
    for row in range(transcripts.shape[0]):
        for col in range(transcripts.shape[1]):
            message = transcripts[row, col].astype('<u4').tobytes()
            digest = StreamDigest(message, key=key).digest()
            if int.from_bytes(digest, 'little') <= training_threshold:
                yield (row, col, digest)

def search_batch(batch: TrainingBatch, training_threshold: int, device: str, cancelled=lambda: False):
    cfg = batch.config
    if batch.native and (cfg.tile_h, cfg.tile_w) == (8, 8):
        from . import native_cuda
        from .gpu_noisy_gemm import TrainingCancelled
        if cancelled():
            raise TrainingCancelled('training job cancelled')
        transcripts = native_cuda.transcripts_device(batch.a_noised, batch.b_noised_t)
        if cancelled():
            raise TrainingCancelled('training job cancelled')
        matches = native_cuda.match_candidates(transcripts, batch.commitment.noise_seed_A, training_threshold)
        if cancelled():
            raise TrainingCancelled('training job cancelled')
        return iter(matches)
    transcripts = gpu_transcripts(batch.a_noised, batch.b_noised, cfg.tile_h, cfg.tile_w, device, cancelled)
    return matching_tiles(transcripts, batch.commitment.noise_seed_A, training_threshold)

def S3_training_threshold(training_threshold: int, config: TrainingConfig, threshold_kind: str='base') -> int:
    if type(training_threshold) is not int or not 0 < training_threshold < 2 ** 256:
        raise ValueError(constant_text(FixedAlias('training.validation.coordinator_acceptance_bound_value')))
    if threshold_kind == 'base':
        return adjusted_training_threshold(training_threshold, config)
    if threshold_kind == 'final':
        return training_threshold
    raise ValueError(constant_text(FixedAlias('training.validation.coordinator_acceptance_bound_kind')))

def verified_S3_submission(batch: TrainingBatch, header, cert_version: int, row: int, col: int, gpu_digest: bytes, training_threshold: int):
    import struct
    if type(cert_version) is not int or cert_version not in (1, 2, 3):
        raise ValueError(constant_text(_VERSION_ERROR))
    if not isinstance(gpu_digest, bytes) or len(gpu_digest) != 32:
        raise ValueError(constant_text(_DIGEST_SIZE_ERROR))
    if type(training_threshold) is not int or not 0 < training_threshold < 2 ** 256:
        raise ValueError(constant_text(FixedAlias('training.validation.final_acceptance_bound_range')))
    if type(row) is not int or type(col) is not int:
        raise ValueError(constant_text(_COORDINATE_ERROR))
    from workrunner import protocol_types as pm
    from S5_base.noisy_gemm import NoisyGemm
    proof = batch.plain_proof(row, col)
    cfg = batch.config
    reference = NoisyGemm(noise_rank=RANK, hash_tile_h=cfg.tile_h, hash_tile_w=cfg.tile_w)
    _, has_full_tiles, transcripts = reference._process_output_tile(batch.a_noised.cpu(), batch.b_noised.cpu(), cfg.k, row * cfg.tile_h, (row + 1) * cfg.tile_h, col * cfg.tile_w, (col + 1) * cfg.tile_w)
    if not has_full_tiles:
        raise RuntimeError(constant_text(_REFERENCE_TILE_ERROR))
    words = [int(value) for value in transcripts[0][0].data]
    message = struct.pack('<16I', *words)
    digest = StreamDigest(message, key=batch.commitment.noise_seed_A).digest()
    if digest != gpu_digest:
        raise RuntimeError(constant_text(_DIGEST_MISMATCH))
    if int.from_bytes(digest, 'little') > training_threshold:
        raise ValueError(constant_text(_SUBMISSION_LIMIT_ERROR))
    valid, message = pm.verify_plain_proof_for_cert_version(cert_version, header, proof, compact_limit_override=_REFERENCE_COMPACT_LIMIT)
    if not valid:
        raise RuntimeError(constant_text(_VERIFICATION_ERROR) + message)
    return (proof, digest)
