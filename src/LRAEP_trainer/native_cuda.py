import ctypes
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
from functools import lru_cache
import numpy as np
import torch
from .paths import cache_dir
try:
    import fcntl
except ImportError:
    fcntl = None
LOG = logging.getLogger('LRAEP_trainer')
SOURCE = Path(__file__).with_name('cuda')
_LIBRARIES = {}
_LOCK = threading.Lock()
_UNAVAILABLE = {}

class NativeUnavailable(RuntimeError):
    pass

def _configure(lib):
    pointer, integer = (ctypes.c_void_p, ctypes.c_int)
    declarations = {'LRAEP_transcripts_i8': [pointer, pointer, integer, integer, integer, pointer, pointer, integer], 'LRAEP_hash_candidates': [pointer, integer, pointer, pointer, pointer, pointer, pointer, integer, pointer], 'LRAEP_noise_factors': [integer, integer, integer, pointer, pointer, pointer, pointer, pointer, pointer, pointer], 'LRAEP_apply_noise': [pointer, pointer, integer, integer, integer, pointer, pointer, pointer, pointer, pointer, pointer, pointer], 'LRAEP_merkle_leaves': [pointer, ctypes.c_size_t, pointer, pointer, pointer], 'LRAEP_merkle_parents': [pointer, integer, pointer, pointer, integer, pointer]}
    for name, arguments in declarations.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = (arguments, integer)
    lib.LRAEP_cuda_error_string.argtypes = [integer]
    lib.LRAEP_cuda_error_string.restype = ctypes.c_char_p
    return lib

def load_native(device='cuda:0'):
    if fcntl is None:
        raise NativeUnavailable('native CUDA builds currently require Linux; the reference path remains available')
    if not torch.cuda.is_available():
        raise NativeUnavailable('CUDA is unavailable')
    destination = torch.device(device)
    if destination.type != 'cuda':
        raise NativeUnavailable('native kernels require CUDA')
    capability = torch.cuda.get_device_capability(destination)
    if capability < (6, 1):
        raise NativeUnavailable('native packed integer kernels require compute capability 6.1+')
    architecture = f'{capability[0]}{capability[1]}'
    with _LOCK:
        if architecture in _LIBRARIES:
            return _LIBRARIES[architecture]
        compiler = shutil.which('nvcc')
        if not compiler:
            raise NativeUnavailable('nvcc is unavailable; install a compatible CUDA toolkit')
        names = ('transcripts.cu', 'hashing.cu', 'preparation.cu', 'merkle.cu', 'runtime.cu', 'blake3.cuh')
        paths = [SOURCE / name for name in names]
        if any((not path.is_file() for path in paths)):
            raise NativeUnavailable('native CUDA source files are incomplete')
        fingerprint = hashlib.sha256(architecture.encode() + os.fsencode(compiler))
        for path in paths:
            fingerprint.update(path.read_bytes())
        cache = cache_dir() / 'native'
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / f'LRAEP-sm{architecture}-{fingerprint.hexdigest()[:16]}.so'
        with (cache / 'build.lock').open('w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not library.exists():
                LOG.info('Building native CUDA kernels for sm_%s (cached after this build)', architecture)
                temporary = library.with_suffix(f'.{os.getpid()}.tmp.so')
                command = [compiler, '-O3', '-std=c++17', '--shared', '-Xcompiler=-fPIC', f'-gencode=arch=compute_{architecture},code=sm_{architecture}', '-I', str(SOURCE), *[str(p) for p in paths if p.suffix == '.cu'], '-o', str(temporary)]
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
                except (OSError, subprocess.SubprocessError) as exc:
                    temporary.unlink(missing_ok=True)
                    raise NativeUnavailable(f'native CUDA build failed: {exc}') from exc
                (cache / 'build.log').write_text(result.stdout + result.stderr)
                if result.returncode:
                    temporary.unlink(missing_ok=True)
                    raise NativeUnavailable(f"native CUDA build failed; see {cache / 'build.log'}")
                temporary.replace(library)
        try:
            lib = _configure(ctypes.CDLL(str(library)))
        except (OSError, AttributeError) as exc:
            raise NativeUnavailable(f'cannot load native CUDA library: {exc}') from exc
        _LIBRARIES[architecture] = lib
        return lib

def available(device='cuda:0') -> bool:
    destination = str(device)
    if destination in _UNAVAILABLE:
        return False
    try:
        load_native(device)
        return True
    except NativeUnavailable as exc:
        _UNAVAILABLE[destination] = str(exc)
        LOG.warning('Native CUDA unavailable: %s; using the PyTorch reference path', exc)
        return False

def _check(lib, code):
    if code:
        reason = lib.LRAEP_cuda_error_string(code).decode('utf-8', 'replace')
        raise RuntimeError(f'native CUDA error: {reason}')

def _stream(device):
    return torch.cuda.current_stream(device).cuda_stream

@lru_cache(maxsize=32)
def _cached_words(value: bytes, device_index: int):
    return torch.from_numpy(np.frombuffer(value, dtype='<u4').copy().view(np.int32)).to(f'cuda:{device_index}')

def _words(value: bytes, device):
    if len(value) != 32:
        raise ValueError('native CUDA keys and targets must contain 32 bytes')
    destination = torch.device(device)
    index = destination.index if destination.index is not None else torch.cuda.current_device()
    words = _cached_words(value, index)
    words.record_stream(torch.cuda.current_stream(destination))
    return words

def _matrix(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_cuda or value.dtype != torch.int8:
        raise ValueError(f'{name} must be a CUDA int8 tensor')
    if value.ndim != 2 or not all(value.shape):
        raise ValueError(f'{name} must be a nonempty matrix')
    value = value.contiguous()
    if value.data_ptr() % 4:
        value = value.clone()
    value.record_stream(torch.cuda.current_stream(value.device))
    return value

@torch.inference_mode()
def merkle_root(signal, key: bytes) -> bytes:
    signal = _matrix(signal, 'signal')
    with torch.cuda.device(signal.device):
        lib, stream = (load_native(signal.device), _stream(signal.device))
        key_gpu = _words(key, signal.device)
        leaves = (signal.numel() + 1023) // 1024
        layer = torch.empty((leaves, 8), dtype=torch.int32, device=signal.device)
        _check(lib, lib.LRAEP_merkle_leaves(signal.data_ptr(), signal.numel(), key_gpu.data_ptr(), layer.data_ptr(), stream))
        while layer.shape[0] > 1:
            count = layer.shape[0]
            parent = torch.empty(((count + 1) // 2, 8), dtype=torch.int32, device=signal.device)
            _check(lib, lib.LRAEP_merkle_parents(layer.data_ptr(), count, key_gpu.data_ptr(), parent.data_ptr(), int(count == 2), stream))
            layer = parent
        return layer.cpu().numpy().astype('<i4', copy=False).tobytes()

@torch.inference_mode()
def add_noise(a, bt, key_a: bytes, key_b: bytes):
    a, bt = (_matrix(a, 'a'), _matrix(bt, 'bt'))
    m, k = a.shape
    n, other_k = bt.shape
    if k != other_k or a.device != bt.device or min(m, n, k) < 128:
        raise ValueError('native noise requires compatible rank-128 signal matrices on one GPU')
    with torch.cuda.device(a.device):
        lib, stream = (load_native(a.device), _stream(a.device))
        ka, kb = (_words(key_a, a.device), _words(key_b, a.device))
        al = torch.empty((m, 128), dtype=torch.int8, device=a.device)
        brt = torch.empty((n, 128), dtype=torch.int8, device=a.device)
        ai = torch.empty((k, 2), dtype=torch.int32, device=a.device)
        bi = torch.empty_like(ai)
        output_a, output_bt = (torch.empty_like(a), torch.empty_like(bt))
        _check(lib, lib.LRAEP_noise_factors(m, n, k, ka.data_ptr(), kb.data_ptr(), al.data_ptr(), brt.data_ptr(), ai.data_ptr(), bi.data_ptr(), stream))
        _check(lib, lib.LRAEP_apply_noise(a.data_ptr(), bt.data_ptr(), m, n, k, al.data_ptr(), brt.data_ptr(), ai.data_ptr(), bi.data_ptr(), output_a.data_ptr(), output_bt.data_ptr(), stream))
        return (output_a, output_bt)

@torch.inference_mode()
def transcripts_device(a, bt, *, variant=0):
    a, bt = (_matrix(a, 'a'), _matrix(bt, 'bt'))
    m, k = a.shape
    n, other_k = bt.shape
    if k != other_k or k % 128 or k > 65536 or m % 8 or n % 8 or (a.device != bt.device):
        raise ValueError('native transcripts require compatible 8x8 tiles and rank-128 slabs')
    with torch.cuda.device(a.device):
        lib, stream = (load_native(a.device), _stream(a.device))
        output = torch.empty((m // 8, n // 8, 16), dtype=torch.int32, device=a.device)
        _check(lib, lib.LRAEP_transcripts_i8(a.data_ptr(), bt.data_ptr(), m, n, k, output.data_ptr(), stream, variant))
        return output

@torch.inference_mode()
def hash_candidates(transcripts, key: bytes, target: int, *, capacity=64):
    if not isinstance(transcripts, torch.Tensor) or not transcripts.is_cuda or transcripts.dtype != torch.int32 or (transcripts.ndim != 3) or (transcripts.shape[-1] != 16) or (type(target) is not int) or (not 0 <= target < 2 ** 256):
        raise ValueError('native hashing requires CUDA int32 transcripts and a 256-bit target')
    if type(capacity) is not int or capacity < 1:
        raise ValueError('candidate capacity must be positive')
    transcripts = transcripts.contiguous()
    rows, cols, _ = transcripts.shape
    count = rows * cols
    if not count:
        return []
    with torch.cuda.device(transcripts.device):
        lib, stream = (load_native(transcripts.device), _stream(transcripts.device))
        key_gpu = _words(key, transcripts.device)
        target_gpu = _words(target.to_bytes(32, 'little'), transcripts.device)
        size = min(capacity, count)
        while True:
            indices = torch.empty(size, dtype=torch.int32, device=transcripts.device)
            digests = torch.empty((size, 32), dtype=torch.uint8, device=transcripts.device)
            matches = torch.empty(1, dtype=torch.int32, device=transcripts.device)
            _check(lib, lib.LRAEP_hash_candidates(transcripts.data_ptr(), count, key_gpu.data_ptr(), target_gpu.data_ptr(), indices.data_ptr(), digests.data_ptr(), matches.data_ptr(), size, stream))
            found = int(matches.cpu()[0])
            if found <= size:
                if not found:
                    return []
                found_indices = indices[:found].cpu().tolist()
                found_digests = digests[:found].cpu().numpy()
                result = [(index // cols, index % cols, digest.tobytes()) for index, digest in zip(found_indices, found_digests)]
                return sorted(result, key=lambda item: item[:2])
            size = min(found, count)
