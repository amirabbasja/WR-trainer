import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading

from .paths import cache_dir
from blake3 import blake3

_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_MASK = 0xFFFFFFFF


_BUILD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _native():
    compiler = shutil.which('c++')
    if compiler is None or os.name != 'posix':
        return None
    source = Path(__file__).with_name('hashing_native.cpp')
    if not source.is_file():
        return None
    try:
        with _BUILD_LOCK:
            folder = cache_dir() / 'hashing'
            folder.mkdir(parents=True, exist_ok=True)
            library = folder / ('tree-' + hashlib.sha256(source.read_bytes()).hexdigest()[:16] + '.so')
            if not library.is_file():
                temporary = library.with_suffix(f'.{os.getpid()}.tmp.so')
                try:
                    subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(temporary)], check=True, capture_output=True, timeout=60)
                    temporary.replace(library)
                finally:
                    temporary.unlink(missing_ok=True)
            binding = ctypes.CDLL(str(library))
            pointer = ctypes.c_void_p
            binding.lraep_hash_chunk.argtypes = [pointer, ctypes.c_uint, pointer, ctypes.c_uint64, ctypes.c_uint, pointer]
            binding.lraep_hash_chunk.restype = None
            binding.lraep_hash_parent.argtypes = [pointer, pointer, pointer, ctypes.c_uint, pointer]
            binding.lraep_hash_parent.restype = None
            return binding
    except (OSError, subprocess.SubprocessError):
        return None


def _rotr(value, count):
    return ((value >> count) | (value << (32 - count))) & _MASK


def compress(block, cv, counter, block_len, flags):
    words = struct.unpack('<16I', block) if isinstance(block, bytes) else tuple(block)
    cv = struct.unpack('<8I', cv) if isinstance(cv, bytes) else tuple(cv)
    state = list(cv) + list(_IV[:4]) + [counter & _MASK, counter >> 32, block_len, flags]

    def mix(a, b, c, d, x, y):
        state[a] = (state[a] + state[b] + x) & _MASK
        state[d] = _rotr(state[d] ^ state[a], 16)
        state[c] = (state[c] + state[d]) & _MASK
        state[b] = _rotr(state[b] ^ state[c], 12)
        state[a] = (state[a] + state[b] + y) & _MASK
        state[d] = _rotr(state[d] ^ state[a], 8)
        state[c] = (state[c] + state[d]) & _MASK
        state[b] = _rotr(state[b] ^ state[c], 7)

    for _ in range(7):
        mix(0, 4, 8, 12, words[0], words[1])
        mix(1, 5, 9, 13, words[2], words[3])
        mix(2, 6, 10, 14, words[4], words[5])
        mix(3, 7, 11, 15, words[6], words[7])
        mix(0, 5, 10, 15, words[8], words[9])
        mix(1, 6, 11, 12, words[10], words[11])
        mix(2, 7, 8, 13, words[12], words[13])
        mix(3, 4, 9, 14, words[14], words[15])
        words = tuple(words[i] for i in _PERM)
    return struct.pack('<16I', *(state[i] ^ state[i + 8] for i in range(8)), *(state[i + 8] ^ cv[i] for i in range(8)))


def chunk_cv(data, key, index=0, root=False):
    if not isinstance(data, bytes) or len(data) > 1024 or len(key) != 32 or type(index) is not int or not 0 <= index < 1 << 64:
        raise ValueError('invalid keyed chunk')
    native = _native()
    if native is not None:
        output = ctypes.create_string_buffer(32)
        native.lraep_hash_chunk(data, len(data), key, index, bool(root), output)
        return output.raw
    cv = key
    blocks = [data[i:i + 64] for i in range(0, len(data), 64)] or [b'']
    for i, block in enumerate(blocks):
        last = i == len(blocks) - 1
        flags = 16 | (1 if i == 0 else 0) | (2 if last else 0) | (8 if last and root else 0)
        cv = compress(block.ljust(64, b'\0'), cv, index, len(block), flags)[:32]
    return cv


def parent_cv(left, right, key, root=False):
    if len(left) != 32 or len(right) != 32 or len(key) != 32:
        raise ValueError('invalid keyed parent')
    native = _native()
    if native is not None:
        output = ctypes.create_string_buffer(32)
        native.lraep_hash_parent(left, right, key, bool(root), output)
        return output.raw
    return compress(left + right, key, 0, 64, 16 | 4 | (8 if root else 0))[:32]
