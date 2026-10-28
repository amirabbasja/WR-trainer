from __future__ import annotations
import struct
from blake3 import blake3
import numpy as np
import torch
from .protocol import TrainingConfig, prepare_signal_batch
from .wr_protocol import WRJob, WRShare
SIGNAL_BASE = 64

def wr_nonce_tail(miner_nonce: bytes, worker_nonce: bytes) -> bytes:
    if any((not isinstance(value, bytes) or len(value) != 8 for value in (miner_nonce, worker_nonce))):
        raise ValueError('PearlHash nonces must each contain exactly eight bytes')
    return bytes((byte % 129 - 64 & 255 for byte in blake3(worker_nonce + miner_nonce).digest()[:16]))

def prepare_wr_batch(job: WRJob, config: TrainingConfig, rng: np.random.Generator, *, device='cpu', native=False):
    if not isinstance(job, WRJob) or job.cert_version != 3:
        raise ValueError('PearlHash training requires an observed generation-3 job')
    if job.worker_nonce is None:
        raise ValueError("PearlHash training requires the pool's eight-byte extranonce")
    miner_nonce = rng.bytes(8)
    tail = wr_nonce_tail(miner_nonce, job.worker_nonce)
    destination = torch.device(device)
    a = torch.full((config.m + 1, config.k), SIGNAL_BASE, dtype=torch.int8, device=destination)
    bt = torch.full((config.n, config.k), SIGNAL_BASE, dtype=torch.int8, device=destination)
    a[-1, -16:] = torch.from_numpy(np.frombuffer(tail, dtype=np.int8).copy()).to(destination)
    from LRAEP_trainer import primitives as pm
    header = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
    batch = prepare_signal_batch(header, 3, config, a, bt, device=device, native=native)
    batch.a_noised = batch.a_noised[:config.m].contiguous()
    batch.miner_nonce = miner_nonce
    batch.worker_nonce = job.worker_nonce
    batch.header_bytes = job.header_bytes
    return batch

def _validate_share(share: WRShare) -> None:
    if not isinstance(share, WRShare):
        raise ValueError('PearlHash verification requires a compact share')
    wr_nonce_tail(share.miner_nonce, share.worker_nonce)
    for field in ('m', 'n', 'k', 'rank', 'tile_h', 'tile_w', 'sb'):
        if type(getattr(share, field)) is not int:
            raise ValueError('PearlHash share dimensions must be integers')
    if not 129 <= share.m < 1 << 31 or not 128 <= share.n < 1 << 31 or (not 2048 <= share.k <= 65536) or share.k % 128 or (share.rank != 128) or (share.sb != 1) or (share.tile_h <= 0) or (share.tile_w <= 0) or share.tile_h % 2 or (share.tile_w % 2) or (not 32 <= share.tile_h * share.tile_w <= 256):
        raise ValueError('Unsupported PearlHash work dimensions or signal mode')
    for indices, count, limit in ((share.a_rows, share.tile_h, share.m - 1), (share.b_cols, share.tile_w, share.n)):
        if len(indices) != count or any((type(index) is not int for index in indices)) or any((not 0 <= index < limit for index in indices)) or any((left >= right for left, right in zip(indices, indices[1:]))):
            raise ValueError('PearlHash tile indices must be sorted, unique and outside the nonce row')

def _constant_root(length: int, key: bytes, tail=b'') -> bytes:
    hasher = blake3(key=key)
    block = bytes([SIGNAL_BASE]) * min(length, 1 << 20)
    remaining = length - len(tail)
    while remaining:
        size = min(remaining, len(block))
        hasher.update(block[:size])
        remaining -= size
    hasher.update(tail)
    hasher.update(bytes(-length % 1024))
    return hasher.digest()

def _uniform_rows(indices, label: bytes, key: bytes) -> np.ndarray:
    raw = b''.join((blake3(struct.pack('<I', row * 4 + draw + 1) + bytes(28) + label, key=key).digest() for row in indices for draw in range(4)))
    return ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) & 63) - 32).reshape(-1, 128)

def _permutation(k: int, label: bytes, key: bytes):
    raw = b''.join((blake3(bytes(4) + struct.pack('<I', draw + 1) + bytes(24) + label, key=key).digest() for draw in range((k + 7) // 8)))
    values = np.frombuffer(raw, dtype='<u4')[:k]
    first = (values & 127).astype(np.int64)
    second = first ^ ((values.astype(np.uint64) * 127 >> 32) + 1).astype(np.int64)
    return (first, second)

def reconstruct_wr_digest(header_bytes: bytes, share: WRShare) -> bytes:
    _validate_share(share)
    if not isinstance(header_bytes, bytes) or len(header_bytes) != 76:
        raise ValueError('PearlHash header must contain exactly 76 bytes')
    from LRAEP_trainer import primitives as pm
    from miner_base.commitment_hash import CommitmentHasher
    from miner_base.noisy_gemm import NoisyGemm
    mining = pm.MiningConfiguration(common_dim=share.k, rank=share.rank, mma_type=pm.MMAType.Int7xInt7ToInt32, rows_pattern=pm.PeriodicPattern.from_list([index - share.a_rows[0] for index in share.a_rows]), cols_pattern=pm.PeriodicPattern.from_list([index - share.b_cols[0] for index in share.b_cols]))
    key = CommitmentHasher.get_key(header_bytes, mining)
    a_root = _constant_root(share.m * share.k, key, wr_nonce_tail(share.miner_nonce, share.worker_nonce))
    b_root = _constant_root(share.n * share.k, key)
    commitment = CommitmentHasher.commitment_hash_from_merkle_roots(a_root, b_root, key, salted_dims=(share.m, share.n))
    a_label, b_label = (b'A_tensor' + bytes(24), b'B_tensor' + bytes(24))
    al = _uniform_rows(share.a_rows, a_label, commitment.noise_seed_A)
    br = _uniform_rows(share.b_cols, b_label, commitment.noise_seed_B)
    ai, aj = _permutation(share.k, a_label, commitment.noise_seed_A)
    bi, bj = _permutation(share.k, b_label, commitment.noise_seed_B)
    a = torch.from_numpy((SIGNAL_BASE + al[:, ai] - al[:, aj]).astype(np.int8))
    b = torch.from_numpy((SIGNAL_BASE + br[:, bi] - br[:, bj]).astype(np.int8).T.copy())
    reference = NoisyGemm(noise_rank=128, hash_tile_h=share.tile_h, hash_tile_w=share.tile_w)
    _, complete, transcripts = reference._process_output_tile(a, b, share.k, 0, share.tile_h, 0, share.tile_w)
    if not complete:
        raise RuntimeError('CPU reference produced no complete PearlHash tile')
    payload = struct.pack('<16I', *(int(word) for word in transcripts[0][0].data))
    return blake3(payload, key=commitment.noise_seed_A).digest()

def verified_wr_share(batch, row: int, col: int, gpu_digest: bytes, final_target: int) -> WRShare:
    cfg = batch.config
    if type(row) is not int or type(col) is not int or (not 0 <= row < cfg.m // cfg.tile_h) or (not 0 <= col < cfg.n // cfg.tile_w):
        raise ValueError('PearlHash tile coordinates are out of range')
    if not isinstance(gpu_digest, bytes) or len(gpu_digest) != 32:
        raise ValueError('PearlHash candidate digest must contain exactly 32 bytes')
    if type(final_target) is not int or not 0 < final_target < 1 << 256:
        raise ValueError('PearlHash final target must be a positive uint256')
    share = WRShare(miner_nonce=batch.miner_nonce, worker_nonce=batch.worker_nonce, m=cfg.m + 1, n=cfg.n, k=cfg.k, rank=128, tile_h=cfg.tile_h, tile_w=cfg.tile_w, a_rows=tuple(range(row * cfg.tile_h, (row + 1) * cfg.tile_h)), b_cols=tuple(range(col * cfg.tile_w, (col + 1) * cfg.tile_w)))
    digest = reconstruct_wr_digest(batch.header_bytes, share)
    if digest != gpu_digest:
        raise RuntimeError('GPU PearlHash digest differs from independently reconstructed CPU work')
    if int.from_bytes(digest, 'little') > final_target:
        raise ValueError('PearlHash candidate does not meet the exact pool target')
    return share
build_wr_share = verified_wr_share
