from __future__ import annotations
from .noise_bounds import tensor_domain, noise_draw
import struct
from .stream_digest import StreamDigest
import numpy as np
import torch
from .protocol import TrainingConfig, prepare_signal_batch
from .training_protocol import TrainingJob, TrainingSubmission
from .fixed_constants import FixedAlias, constant_text

_BAD_SEED_SIZE = 'S7 session seeds must each contain exactly eight bytes'
_MISSING_SERVER_SEED = "S7 training requires the S3's eight-byte server seed"
_MISSING_RESULT = FixedAlias('training.validation.compact_attestation_required')
_BAD_DIMENSIONS = FixedAlias('training.validation.batch_submission_dimensions_type')
_BAD_TILE_INDICES = 'S7 tile indices must be sorted, unique and outside the seed row'
_INVALID_TRAINING_THRESHOLD = FixedAlias('training.validation.batch_acceptance_bound_range')
_TRAINING_THRESHOLD_EXCEEDED = FixedAlias('training.validation.batch_candidate_outside_acceptance_bound')



def signal_base():
    return int('40', 16)

def wr_seed_tail(local_seed: bytes, device_seed: bytes) -> bytes:
    if any((not isinstance(value, bytes) or len(value) != 8 for value in (local_seed, device_seed))):
        raise ValueError(_BAD_SEED_SIZE)
    return bytes((byte % 129 - 64 & 255 for byte in StreamDigest(device_seed + local_seed).digest()[:16]))

def prepare_training_batch(job: TrainingJob, config: TrainingConfig, rng: np.random.Generator, *, device='cpu', native=False):
    if not isinstance(job, TrainingJob) or job.cert_version != 3:
        raise ValueError(constant_text(FixedAlias('training.validation.batch_generation_three_required')))
    if job.device_seed is None:
        raise ValueError(_MISSING_SERVER_SEED)
    local_seed = rng.bytes(8)
    tail = wr_seed_tail(local_seed, job.device_seed)
    destination = torch.device(device)
    a = torch.full((config.m + 1, config.k), signal_base(), dtype=torch.int8, device=destination)
    bt = torch.full((config.n, config.k), signal_base(), dtype=torch.int8, device=destination)
    a[-1, -16:] = torch.from_numpy(np.frombuffer(tail, dtype=np.int8).copy()).to(destination)
    from workrunner import protocol_types as pm
    header = pm.PartialHeader.from_bytes(job.header_bytes)
    batch = prepare_signal_batch(header, 3, config, a, bt, device=device, native=native)
    batch.a_noised = batch.a_noised[:config.m].contiguous()
    batch.local_seed = local_seed
    batch.device_seed = job.device_seed
    batch.header_bytes = job.header_bytes
    return batch

def _validate_submission(batch_result: TrainingSubmission) -> None:
    if not isinstance(batch_result, TrainingSubmission):
        raise ValueError(constant_text(_MISSING_RESULT))
    wr_seed_tail(batch_result.local_seed, batch_result.device_seed)
    for field in ('m', 'n', 'k', 'rank', 'tile_h', 'tile_w', 'sb'):
        if type(getattr(batch_result, field)) is not int:
            raise ValueError(constant_text(_BAD_DIMENSIONS))
    if not 129 <= batch_result.m < 1 << 31 or not 128 <= batch_result.n < 1 << 31 or (not 2048 <= batch_result.k <= 65536) or batch_result.k % 128 or (batch_result.rank != 128) or (batch_result.sb != 1) or (batch_result.tile_h <= 0) or (batch_result.tile_w <= 0) or batch_result.tile_h % 2 or (batch_result.tile_w % 2) or (not 32 <= batch_result.tile_h * batch_result.tile_w <= 256):
        raise ValueError(constant_text(FixedAlias('training.validation.batch_dimensions_or_signal_mode')))
    for indices, count, limit in ((batch_result.a_rows, batch_result.tile_h, batch_result.m - 1), (batch_result.b_cols, batch_result.tile_w, batch_result.n)):
        if len(indices) != count or any((type(index) is not int for index in indices)) or any((not 0 <= index < limit for index in indices)) or any((left >= right for left, right in zip(indices, indices[1:]))):
            raise ValueError(_BAD_TILE_INDICES)

def _constant_root(length: int, key: bytes, tail=b'') -> bytes:
    hasher = StreamDigest(key=key)
    block = bytes([signal_base()]) * min(length, 1 << 20)
    remaining = length - len(tail)
    while remaining:
        size = min(remaining, len(block))
        hasher.update(block[:size])
        remaining -= size
    hasher.update(tail)
    hasher.update(bytes(-length % 1024))
    return hasher.digest()

def _uniform_rows(indices, label: bytes, key: bytes) -> np.ndarray:
    raw = b''.join((noise_draw(StreamDigest, row * 4 + draw, label, key, 0) for row in indices for draw in range(4)))
    return ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) & 63) - 32).reshape(-1, 128)

def _permutation(k: int, label: bytes, key: bytes):
    raw = b''.join((noise_draw(StreamDigest, draw, label, key, 1) for draw in range((k + 7) // 8)))
    values = np.frombuffer(raw, dtype='<u4')[:k]
    first = (values & 127).astype(np.int64)
    second = first ^ ((values.astype(np.uint64) * 127 >> 32) + 1).astype(np.int64)
    return (first, second)

def reconstruct_wr_digest(header_bytes: bytes, batch_result: TrainingSubmission) -> bytes:
    _validate_submission(batch_result)
    if not isinstance(header_bytes, bytes) or len(header_bytes) != 76:
        raise ValueError(constant_text(FixedAlias('training.validation.batch_header_length')))
    from workrunner import protocol_types as pm
    from S5_base.commitment_hash import CommitmentHasher
    from S5_base.noisy_gemm import NoisyGemm
    S1 = pm.S1Configuration(common_dim=batch_result.k, rank=batch_result.rank, mma_type=pm.MMAType.Int7xInt7ToInt32, rows_pattern=pm.PeriodicPattern.from_list([index - batch_result.a_rows[0] for index in batch_result.a_rows]), cols_pattern=pm.PeriodicPattern.from_list([index - batch_result.b_cols[0] for index in batch_result.b_cols]))
    key = CommitmentHasher.get_key(header_bytes, S1)
    a_root = _constant_root(batch_result.m * batch_result.k, key, wr_seed_tail(batch_result.local_seed, batch_result.device_seed))
    b_root = _constant_root(batch_result.n * batch_result.k, key)
    commitment = CommitmentHasher.commitment_hash_from_tree_roots(a_root, b_root, key, salted_dims=(batch_result.m, batch_result.n))
    with tensor_domain() as label:
        al = _uniform_rows(batch_result.a_rows, label, commitment.noise_seed_A)
        ai, aj = _permutation(batch_result.k, label, commitment.noise_seed_A)
    with tensor_domain(True) as label:
        br = _uniform_rows(batch_result.b_cols, label, commitment.noise_seed_B)
        bi, bj = _permutation(batch_result.k, label, commitment.noise_seed_B)
    a = torch.from_numpy((signal_base() + al[:, ai] - al[:, aj]).astype(np.int8))
    b = torch.from_numpy((signal_base() + br[:, bi] - br[:, bj]).astype(np.int8).T.copy())
    reference = NoisyGemm(noise_rank=128, hash_tile_h=batch_result.tile_h, hash_tile_w=batch_result.tile_w)
    _, complete, transcripts = reference._process_output_tile(a, b, batch_result.k, 0, batch_result.tile_h, 0, batch_result.tile_w)
    if not complete:
        raise RuntimeError(constant_text(FixedAlias('training.validation.cpu_batch_tile_incomplete')))
    payload = struct.pack('<16I', *(int(word) for word in transcripts[0][0].data))
    return StreamDigest(payload, key=commitment.noise_seed_A).digest()

def verified_training_submission(batch, row: int, col: int, gpu_digest: bytes, training_threshold: int) -> TrainingSubmission:
    """Check the training threshold and independently reconstruct the batch result."""
    training_config = batch.config
    if type(row) is not int or type(col) is not int or (not 0 <= row < training_config.m // training_config.tile_h) or (not 0 <= col < training_config.n // training_config.tile_w):
        raise ValueError(constant_text(FixedAlias('training.validation.batch_tile_coordinate_range')))
    if not isinstance(gpu_digest, bytes) or len(gpu_digest) != 32:
        raise ValueError(constant_text(FixedAlias('training.validation.batch_candidate_digest_length')))
    if type(training_threshold) is not int or not 0 < training_threshold < 1 << 256:
        raise ValueError(constant_text(_INVALID_TRAINING_THRESHOLD))
    batch_result = TrainingSubmission(local_seed=batch.local_seed, device_seed=batch.device_seed, m=training_config.m + 1, n=training_config.n, k=training_config.k, rank=128, tile_h=training_config.tile_h, tile_w=training_config.tile_w, a_rows=tuple(range(row * training_config.tile_h, (row + 1) * training_config.tile_h)), b_cols=tuple(range(col * training_config.tile_w, (col + 1) * training_config.tile_w)))
    digest = reconstruct_wr_digest(batch.header_bytes, batch_result)
    if digest != gpu_digest:
        raise RuntimeError(constant_text(FixedAlias('training.validation.accelerator_batch_digest_mismatch')))
    if int.from_bytes(digest, 'little') > training_threshold:
        raise ValueError(constant_text(_TRAINING_THRESHOLD_EXCEEDED))
    return batch_result
build_wr_submission = verified_training_submission
