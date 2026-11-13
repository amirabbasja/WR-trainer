import ctypes
from contextlib import contextmanager
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
import numpy as np
import torch
from .paths import cache_dir
from .training_artifacts import artifacts_enabled
try:
    import fcntl
except ImportError:
    fcntl = None
LOG = logging.getLogger('workrunner')
SOURCE = Path(__file__).with_name('cuda')
_LIBRARIES = {}
_LOCK = threading.Lock()
_UNAVAILABLE = {}

class NativeUnavailable(RuntimeError):
    pass

def _configure(lib):
    pointer, integer = (ctypes.c_void_p, ctypes.c_int)
    declarations = {'wr_transcripts_i8': [pointer, pointer, integer, integer, integer, pointer, pointer, integer], 'wr_match_candidates': [pointer, integer, pointer, pointer, pointer, pointer, pointer, integer, pointer], 'wr_noise_factors': [integer, integer, integer, pointer, pointer, pointer, pointer, pointer, pointer, pointer], 'wr_apply_noise': [pointer, pointer, integer, integer, integer, pointer, pointer, pointer, pointer, pointer, pointer, pointer], 'wr_tree_leaves': [pointer, ctypes.c_size_t, pointer, pointer, pointer], 'wr_tree_parents': [pointer, integer, pointer, pointer, integer, pointer]}
    for name, arguments in declarations.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = (arguments, integer)
    lib.wr_cuda_error_string.argtypes = [integer]
    lib.wr_cuda_error_string.restype = ctypes.c_char_p
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
        names = ('transcripts.cu', 'hashing.cu', 'preparation.cu', 'tree.cu', 'runtime.cu', 'digest_core.cuh')
        paths = [SOURCE / name for name in names]
        if any((not path.is_file() for path in paths)):
            raise NativeUnavailable('native CUDA source files are incomplete')
        fingerprint = hashlib.sha256(architecture.encode() + os.fsencode(compiler))
        for path in paths:
            fingerprint.update(path.read_bytes())
        cache = cache_dir() / 'native'
        library = cache / f'WR-sm{architecture}-{fingerprint.hexdigest()[:16]}.so'
        if not library.is_file():
            if not artifacts_enabled():
                raise NativeUnavailable('native CUDA library is unavailable in the cache')
            _build_library(library, compiler, architecture, paths)
        try:
            lib = _configure(ctypes.CDLL(str(library)))
        except (OSError, AttributeError) as exc:
            raise NativeUnavailable(f'cannot load native CUDA library: {exc}') from exc
        _LIBRARIES[architecture] = lib
        return lib

def _build_library(library, compiler, architecture, paths):
    cache = library.parent
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'build.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if library.is_file():
            return
        LOG.info('Building native CUDA kernels for sm_%s (cached after this build)', architecture)
        temporary = library.with_suffix(f'.{os.getpid()}.tmp.so')
        command = [compiler, '-O3', '-std=c++17', '--shared', '-Xcompiler=-fPIC', f'-gencode=arch=compute_{architecture},code=sm_{architecture}', '-I', str(SOURCE), *[str(p) for p in paths if p.suffix == '.cu'], '-o', str(temporary)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            (cache / 'build.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                raise NativeUnavailable(f"native CUDA build failed; see {cache / 'build.log'}")
            temporary.replace(library)
        except (OSError, subprocess.SubprocessError) as exc:
            raise NativeUnavailable(f'native CUDA build failed: {exc}') from exc
        finally:
            temporary.unlink(missing_ok=True)

def available(device='cuda:0') -> bool:
    destination = str(device)
    if destination in _UNAVAILABLE:
        return False
    try:
        load_native(device)
        return True
    except NativeUnavailable as exc:
        if artifacts_enabled():
            _UNAVAILABLE[destination] = str(exc)
            LOG.warning('Native CUDA unavailable: %s; using the PyTorch reference path', exc)
        return False

def _check(lib, code):
    if code:
        reason = lib.wr_cuda_error_string(code).decode('utf-8', 'replace')
        raise RuntimeError(f'native CUDA error: {reason}')

def _stream(device):
    return torch.cuda.current_stream(device).cuda_stream

def _wipe_words(buffers, stream, original_error=None):
    if not buffers:
        return
    failure = None
    words = None
    try:
        try:
            with torch.cuda.stream(stream):
                for words in buffers:
                    try:
                        words.zero_()
                    except Exception as error:
                        if failure is None:
                            failure = error
        except Exception as error:
            if failure is None:
                failure = error
        try:
            stream.synchronize()
        except Exception as error:
            if failure is None:
                failure = error
        if failure is not None:
            if original_error is not None:
                original_error.add_note('Temporary CUDA work buffers could not be fully cleared.')
            else:
                raise RuntimeError('Could not clear temporary CUDA work buffers') from failure
    finally:
        buffers.clear()
        del words, original_error


@contextmanager
def _temporary_words(device, *values):
    stream = torch.cuda.current_stream(device)
    buffers = []
    original_error = None
    value = None
    try:
        for value in values:
            buffers.append(_words(value, device))
        value = None
        values = ()
        yield buffers
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            _wipe_words(buffers, stream, original_error)
        finally:
            buffers.clear()
            del values, value, original_error


def _words(value: bytes, device):
    if len(value) != 32:
        raise ValueError('native CUDA keys and targets must contain 32 bytes')
    destination = torch.device(device)
    if destination.type != 'cuda':
        raise ValueError('native CUDA words require a CUDA device')
    host = np.frombuffer(value, dtype='<u4').copy().view(np.int32)
    words = None
    stream = None
    try:
        stream = torch.cuda.current_stream(destination)
        words = torch.from_numpy(host).to(destination, non_blocking=False)
        words.record_stream(stream)
        return words
    except BaseException as error:
        if words is not None:
            _wipe_words([words], stream, error)
        raise
    finally:
        host.fill(0)
        del value, host

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
def tree_root(signal, key: bytes) -> bytes:
    signal = _matrix(signal, 'signal')
    with torch.cuda.device(signal.device):
        lib, stream = (load_native(signal.device), _stream(signal.device))
        with _temporary_words(signal.device, key) as (key_gpu,):
            leaves = (signal.numel() + 1023) // 1024
            layer = torch.empty((leaves, 8), dtype=torch.int32, device=signal.device)
            _check(lib, lib.wr_tree_leaves(signal.data_ptr(), signal.numel(), key_gpu.data_ptr(), layer.data_ptr(), stream))
            while layer.shape[0] > 1:
                count = layer.shape[0]
                parent = torch.empty(((count + 1) // 2, 8), dtype=torch.int32, device=signal.device)
                _check(lib, lib.wr_tree_parents(layer.data_ptr(), count, key_gpu.data_ptr(), parent.data_ptr(), int(count == 2), stream))
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
        al = torch.empty((m, 128), dtype=torch.int8, device=a.device)
        brt = torch.empty((n, 128), dtype=torch.int8, device=a.device)
        ai = torch.empty((k, 2), dtype=torch.int32, device=a.device)
        bi = torch.empty_like(ai)
        output_a, output_bt = (torch.empty_like(a), torch.empty_like(bt))
        with _temporary_words(a.device, key_a, key_b) as (ka, kb):
            _check(lib, lib.wr_noise_factors(m, n, k, ka.data_ptr(), kb.data_ptr(), al.data_ptr(), brt.data_ptr(), ai.data_ptr(), bi.data_ptr(), stream))
        _check(lib, lib.wr_apply_noise(a.data_ptr(), bt.data_ptr(), m, n, k, al.data_ptr(), brt.data_ptr(), ai.data_ptr(), bi.data_ptr(), output_a.data_ptr(), output_bt.data_ptr(), stream))
        return (output_a, output_bt)

@torch.inference_mode()
def transcripts_device(a, bt, *, variant=0, tile_w=8):
    a, bt = (_matrix(a, 'a'), _matrix(bt, 'bt'))
    m, k = a.shape
    n, other_k = bt.shape
    if tile_w == 16 and variant == 0:
        variant = 8
    if (tile_w not in (8, 16) or (variant == 8) != (tile_w == 16)
            or k != other_k or k % 128 or k > 65536 or m % 8 or n % tile_w or a.device != bt.device):
        raise ValueError('native transcripts require compatible 8x8 or 8x16 tiles and rank-128 slabs')
    with torch.cuda.device(a.device):
        lib, stream = (load_native(a.device), _stream(a.device))
        output = torch.empty((m // 8, n // tile_w, 16), dtype=torch.int32, device=a.device)
        _check(lib, lib.wr_transcripts_i8(a.data_ptr(), bt.data_ptr(), m, n, k, output.data_ptr(), stream, variant))
        return output

@torch.inference_mode()
def match_candidates(transcripts, key: bytes, training_threshold: int, *, capacity=64):
    if not isinstance(transcripts, torch.Tensor) or not transcripts.is_cuda or transcripts.dtype != torch.int32 or (transcripts.ndim != 3) or (transcripts.shape[-1] != 16) or (type(training_threshold) is not int) or (not 0 <= training_threshold < 2 ** 256):
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
        with _temporary_words(transcripts.device, key, training_threshold.to_bytes(32, 'little')) as (key_gpu, training_threshold_gpu):
            size = min(capacity, count)
            while True:
                indices = torch.empty(size, dtype=torch.int32, device=transcripts.device)
                digests = torch.empty((size, 32), dtype=torch.uint8, device=transcripts.device)
                matches = torch.empty(1, dtype=torch.int32, device=transcripts.device)
                _check(lib, lib.wr_match_candidates(transcripts.data_ptr(), count, key_gpu.data_ptr(), training_threshold_gpu.data_ptr(), indices.data_ptr(), digests.data_ptr(), matches.data_ptr(), size, stream))
                found = int(matches.cpu()[0])
                if found <= size:
                    break
                size = min(found, count)
        if not found:
            return []
        found_indices = indices[:found].cpu().tolist()
        found_digests = digests[:found].cpu().numpy()
        result = [(index // cols, index % cols, digest.tobytes())
                  for index, digest in zip(found_indices, found_digests)]
        return sorted(result, key=lambda item: item[:2])
