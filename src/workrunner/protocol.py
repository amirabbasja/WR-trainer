import os
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .stream_digest import StreamDigest
from . import OFFICIAL_COMMIT
from .gpu import RANK, gpu_add_sparse_noise, gpu_transcripts
from .noise import generate_noise_factors, generate_sparse_noise_factors
from .helpers import verify_bundle
from .paths import runtime_setting
from .fixed_constants import constant_text

_REFERENCE_COMPACT_LIMIT = 0x207FFFFF
_ROOT_MISMATCH = b'$;Q*p(ZYw|_wtA8!qv>n!iUw!$kT_z&cVUL)zpW<)yT?+)6c?($iczR#LmXchvfJ2htt)<!T'
_VERSION_ERROR = b'(#h1)*Vo9^)561t^pS_e!qwBx!Oq0S)53?*!qwEy$jOJn$km7Rl!w&M#?`_'
_DIGEST_SIZE_ERROR = b';P>)})X&D%!iU4oz{1qihsx5_(}%^vhr-{+#M8^(hm@6v#op7x)B'
_COORDINATE_ERROR = b'^puCx&&Jilhttl>!iU7j$koHn$;Q*d)Q8H_)YFH>!iUbu)55^Q)zk'
_REFERENCE_TILE_ERROR = b'$iczR#LmXchvfJ2ht<Nt!qvjb#KMQy)yTur#KOae$;gMq$jaBt!qdWs)X&D%!iUq&%fb'
_DIGEST_MISMATCH = b';P>)})X&D%!iU4oz{1qihr`an!NS$lhr!jz%7@d>!iUJg!Oq0a#><D~_wt9;!okAT!pX$K'
_SUBMISSION_LIMIT_ERROR = b')78e#$<E2Zht$u;)xw9v$imcz$;i`(%EH3ahtto(hr-{+#M6iLl!w#C)xg5j'
_VERIFICATION_ERROR = b'^puCx&&Jilht<N(!o<_U!-vJ*hseRf&cx2f%ZJgz)y~1r#KzOk$jP0D'

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
            raise ValueError(constant_text(b')WXZc#M8pVhu77}$iauz$k5b>#>vBn#K_Ci%E{D+!r#Qg!o!F3k$}{PlZWcg<%iJ7)y%@xiq6T`($j~_&dScy'))

    def official_config(self):
        from workrunner import primitives as pm
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
        from workrunner import primitives as pm
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
                raise RuntimeError(constant_text(_ROOT_MISMATCH))
            self._tree = tree
        return self._tree

    def leaf_indices_from_rows(self, rows):
        from workrunner import primitives as pm
        return pm.DigestTree.compute_leaf_indices_from_rows(rows, tuple(self.signal.shape))

    def get_multileaf_proof(self, indices):
        return self._materialize().get_multileaf_proof(indices)

def prepare_batch(header, cert_version: int, config: TrainingConfig, rng: np.random.Generator, *, device: str | torch.device='cpu', gpu_signals: bool=False, native: bool=False) -> TrainingBatch:
    if cert_version not in (1, 2, 3):
        raise ValueError(f"{constant_text(b'(#h1)*Vo9^)561t^pS_e!qwBx!Oq0S)53?*!qwEy$jOH')}{cert_version}")
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
        raise ValueError(f"{constant_text(b'(#h1)*Vo9^)561t^pS_e!qwBx!Oq0S)53?*!qwEy$jOH')}{cert_version}")
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

def adjusted_target(base_target: int, config: TrainingConfig) -> int:
    from workrunner import primitives as pm
    if not 0 < base_target < 2 ** 256:
        raise ValueError('node target must be a positive 256-bit integer')
    result = pm.penalized_target_bound(base_target, config.official_config())
    if result is None:
        raise ValueError('target scaling overflows; choose a harder test-network target')
    return result

def matching_tiles(transcripts: np.ndarray, key: bytes, target: int):
    if len(key) != 32 or not 0 <= target < 2 ** 256:
        raise ValueError('invalid transcript hash key or target')
    for row in range(transcripts.shape[0]):
        for col in range(transcripts.shape[1]):
            message = transcripts[row, col].astype('<u4').tobytes()
            digest = StreamDigest(message, key=key).digest()
            if int.from_bytes(digest, 'little') <= target:
                yield (row, col, digest)

def search_batch(batch: TrainingBatch, target: int, device: str, cancelled=lambda: False):
    cfg = batch.config
    if batch.native and (cfg.tile_h, cfg.tile_w) == (8, 8):
        from . import native_cuda
        from .gpu import TrainingCancelled
        if cancelled():
            raise TrainingCancelled('training job cancelled')
        transcripts = native_cuda.transcripts_device(batch.a_noised, batch.b_noised_t)
        if cancelled():
            raise TrainingCancelled('training job cancelled')
        matches = native_cuda.match_candidates(transcripts, batch.commitment.noise_seed_A, target)
        if cancelled():
            raise TrainingCancelled('training job cancelled')
        return iter(matches)
    transcripts = gpu_transcripts(batch.a_noised, batch.b_noised, cfg.tile_h, cfg.tile_w, device, cancelled)
    return matching_tiles(transcripts, batch.commitment.noise_seed_A, target)

def S3_submission_target(target: int, config: TrainingConfig, target_kind: str='base') -> int:
    if type(target) is not int or not 0 < target < 2 ** 256:
        raise ValueError(constant_text(b'^puCw#?`>W(}&8^)YFH>!iUC(*T~e))6UVthn13%ip9>;htA2<!ob4S'))
    if target_kind == 'base':
        return adjusted_target(target, config)
    if target_kind == 'final':
        return target
    raise ValueError(constant_text(b'^puCw#?`>W(}&E?$-{@r($v$3#lnZh#?-=x$km6z&dJ8h'))

def verified_S3_submission(batch: TrainingBatch, header, cert_version: int, row: int, col: int, gpu_digest: bytes, final_target: int):
    import struct
    if type(cert_version) is not int or cert_version not in (1, 2, 3):
        raise ValueError(constant_text(_VERSION_ERROR))
    if not isinstance(gpu_digest, bytes) or len(gpu_digest) != 32:
        raise ValueError(constant_text(_DIGEST_SIZE_ERROR))
    if type(final_target) is not int or not 0 < final_target < 2 ** 256:
        raise ValueError(constant_text(b'!OqFX%ZK!ohttN@z{1mq%F@)+hsDB&#)sF))Xvk+(ZYw7l97tV&eMm^$<xBX!qo'))
    if type(row) is not int or type(col) is not int:
        raise ValueError(constant_text(_COORDINATE_ERROR))
    from workrunner import primitives as pm
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
    if int.from_bytes(digest, 'little') > final_target:
        raise ValueError(constant_text(_SUBMISSION_LIMIT_ERROR))
    valid, message = pm.verify_plain_proof_for_cert_version(cert_version, header, proof, compact_limit_override=_REFERENCE_COMPACT_LIMIT)
    if not valid:
        raise RuntimeError(constant_text(_VERIFICATION_ERROR) + message)
    return (proof, digest)
