from __future__ import annotations
from .noise_domains import tensor_domain, noise_draw
import struct
from .stream_digest import StreamDigest
import numpy as np
import torch
from .protocol import TrainingConfig, prepare_signal_batch
from .wr_protocol import WRJob, WRSubmission
from .fixed_constants import constant_text

_BAD_SEED_SIZE = b'^pJ<i$jQXQ)Q8H_)YFH;#>CHu#K_6h#?Hxy!r#Wk)63q6!p^|Y(}%_0)56p'
_MISSING_SERVER_SEED = b'^pJ<s)yB@r&dI=s)xy@&&eg)yhtto(hxC+y)Q7^(z|Yf)#op7xhr-{})yB!l$;83'
_MISSING_RESULT = b'^pJ<q!qv{f&cw#k&dAA!)xy@&&eg)yhsKA*$jaBo#M6h=&&Jil'
_BAD_DIMENSIONS = b'^pJ<t&&Jilhr`aw!pYRm$jQ`)%F@)+hsDB&&dJllz{1tk'
_BAD_TILE_INDICES = b'^pJ<s&db7w&dI~h#KP2v%F@)+hsDB&)X3G-!o!P)(#g)&(!z(v$-{@p($mz=!@`Ht&%%ev$jQXQht<f?'
_BAD_FINAL_THRESHOLD = b'^pJ<a&dJ8hhttN@z{1mq%F@)+hsDB&#)sF))Xvk+(ZYw)&dJl2l92'
_THRESHOLD_MISMATCH = b'^pJ<d#>vCZ!^YFXhr`Ii)Q8E)(}&8!!qbP-&%%eo-^Rq#hxC+()5g`n!qW'



def signal_base():
    return int('40', 16)

def wr_seed_tail(local_seed: bytes, device_seed: bytes) -> bytes:
    if any((not isinstance(value, bytes) or len(value) != 8 for value in (local_seed, device_seed))):
        raise ValueError(constant_text(_BAD_SEED_SIZE))
    return bytes((byte % 129 - 64 & 255 for byte in StreamDigest(device_seed + local_seed).digest()[:16]))

def prepare_wr_batch(job: WRJob, config: TrainingConfig, rng: np.random.Generator, *, device='cpu', native=False):
    if not isinstance(job, WRJob) or job.cert_version != 3:
        raise ValueError(constant_text(b'^pJ<s)yB@r&dI=s)xy@&&eg)yhsMc=$i>vc)zQMkhrq(g!qvvp&dAA%l!wj8#Q'))
    if job.device_seed is None:
        raise ValueError(constant_text(_MISSING_SERVER_SEED))
    local_seed = rng.bytes(8)
    tail = wr_seed_tail(local_seed, job.device_seed)
    destination = torch.device(device)
    a = torch.full((config.m + 1, config.k), signal_base(), dtype=torch.int8, device=destination)
    bt = torch.full((config.n, config.k), signal_base(), dtype=torch.int8, device=destination)
    a[-1, -16:] = torch.from_numpy(np.frombuffer(tail, dtype=np.int8).copy()).to(destination)
    from workrunner import primitives as pm
    header = pm.PartialHeader.from_bytes(job.header_bytes)
    batch = prepare_signal_batch(header, 3, config, a, bt, device=device, native=native)
    batch.a_noised = batch.a_noised[:config.m].contiguous()
    batch.local_seed = local_seed
    batch.device_seed = job.device_seed
    batch.header_bytes = job.header_bytes
    return batch

def _validate_submission(submission: WRSubmission) -> None:
    if not isinstance(submission, WRSubmission):
        raise ValueError(constant_text(_MISSING_RESULT))
    wr_seed_tail(submission.local_seed, submission.device_seed)
    for field in ('m', 'n', 'k', 'rank', 'tile_h', 'tile_w', 'sb'):
        if type(getattr(submission, field)) is not int:
            raise ValueError(constant_text(_BAD_DIMENSIONS))
    if not 129 <= submission.m < 1 << 31 or not 128 <= submission.n < 1 << 31 or (not 2048 <= submission.k <= 65536) or submission.k % 128 or (submission.rank != 128) or (submission.sb != 1) or (submission.tile_h <= 0) or (submission.tile_w <= 0) or submission.tile_h % 2 or (submission.tile_w % 2) or (not 32 <= submission.tile_h * submission.tile_w <= 256):
        raise ValueError(constant_text(b'^2yZF*Vo9^)561t^pJ<p$koh;!_LaW$<)rs$<&9))rZv1z{$qThswyq!T'))
    for indices, count, limit in ((submission.a_rows, submission.tile_h, submission.m - 1), (submission.b_cols, submission.tile_w, submission.n)):
        if len(indices) != count or any((type(index) is not int for index in indices)) or any((not 0 <= index < limit for index in indices)) or any((left >= right for left, right in zip(indices, indices[1:]))):
            raise ValueError(constant_text(_BAD_TILE_INDICES))

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

def reconstruct_wr_digest(header_bytes: bytes, submission: WRSubmission) -> bytes:
    _validate_submission(submission)
    if not isinstance(header_bytes, bytes) or len(header_bytes) != 76:
        raise ValueError(constant_text(b'^pJ<o!p6hG)rZQ`)YFH=$jQ^j&dG<u-^Rq#%if2Ok%z_J)56p'))
    from workrunner import primitives as pm
    from S5_base.commitment_hash import CommitmentHasher
    from S5_base.noisy_gemm import NoisyGemm
    S1 = pm.S1Configuration(common_dim=submission.k, rank=submission.rank, mma_type=pm.MMAType.Int7xInt7ToInt32, rows_pattern=pm.PeriodicPattern.from_list([index - submission.a_rows[0] for index in submission.a_rows]), cols_pattern=pm.PeriodicPattern.from_list([index - submission.b_cols[0] for index in submission.b_cols]))
    key = CommitmentHasher.get_key(header_bytes, S1)
    a_root = _constant_root(submission.m * submission.k, key, wr_seed_tail(submission.local_seed, submission.device_seed))
    b_root = _constant_root(submission.n * submission.k, key)
    commitment = CommitmentHasher.commitment_hash_from_tree_roots(a_root, b_root, key, salted_dims=(submission.m, submission.n))
    with tensor_domain() as label:
        al = _uniform_rows(submission.a_rows, label, commitment.noise_seed_A)
        ai, aj = _permutation(submission.k, label, commitment.noise_seed_A)
    with tensor_domain(True) as label:
        br = _uniform_rows(submission.b_cols, label, commitment.noise_seed_B)
        bi, bj = _permutation(submission.k, label, commitment.noise_seed_B)
    a = torch.from_numpy((signal_base() + al[:, ai] - al[:, aj]).astype(np.int8))
    b = torch.from_numpy((signal_base() + br[:, bi] - br[:, bj]).astype(np.int8).T.copy())
    reference = NoisyGemm(noise_rank=128, hash_tile_h=submission.tile_h, hash_tile_w=submission.tile_w)
    _, complete, transcripts = reference._process_output_tile(a, b, submission.k, 0, submission.tile_h, 0, submission.tile_w)
    if not complete:
        raise RuntimeError(constant_text(b'<oEK2)xyET)xycd!iU$@$ivda!o!Ej$cMzp%Gb-n)53@JkcZRG%fb'))
    payload = struct.pack('<16I', *(int(word) for word in transcripts[0][0].data))
    return StreamDigest(payload, key=commitment.noise_seed_A).digest()

def verified_wr_submission(batch, row: int, col: int, gpu_digest: bytes, final_target: int) -> WRSubmission:
    cfg = batch.config
    if type(row) is not int or type(col) is not int or (not 0 <= row < cfg.m // cfg.tile_h) or (not 0 <= col < cfg.n // cfg.tile_w):
        raise ValueError(constant_text(b'^pJ<s&db7w#K_3i!_LXZ)56q;#?``y$kNk?$iauz#>v3K'))
    if not isinstance(gpu_digest, bytes) or len(gpu_digest) != 32:
        raise ValueError(constant_text(b'^pJ<d#>vCZ!^YFXhr`am!qn4;%F@)+hs4Oq)5gxphr-{+#M8^(hm@6v#op7x)B'))
    if type(final_target) is not int or not 0 < final_target < 1 << 256:
        raise ValueError(constant_text(_BAD_FINAL_THRESHOLD))
    submission = WRSubmission(local_seed=batch.local_seed, device_seed=batch.device_seed, m=cfg.m + 1, n=cfg.n, k=cfg.k, rank=128, tile_h=cfg.tile_h, tile_w=cfg.tile_w, a_rows=tuple(range(row * cfg.tile_h, (row + 1) * cfg.tile_h)), b_cols=tuple(range(col * cfg.tile_w, (col + 1) * cfg.tile_w)))
    digest = reconstruct_wr_digest(batch.header_bytes, submission)
    if digest != gpu_digest:
        raise RuntimeError(constant_text(b';P>)}^pJ<c&cMRd(}%;(!NJ1S)Q7><$jXP#$-~0e!pXzJ$<xc;ht<Nw$jQ{x)zZY%!o!E;_wt9($kog'))
    if int.from_bytes(digest, 'little') > final_target:
        raise ValueError(constant_text(_THRESHOLD_MISMATCH))
    return submission
build_wr_submission = verified_wr_submission
