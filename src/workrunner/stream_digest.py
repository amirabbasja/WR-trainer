import ctypes
from functools import lru_cache
import hashlib
import operator
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import threading

from .paths import cache_dir
from .string_codec import customDecoder
from .training_artifacts import artifacts_enabled

_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
_IV_BYTES = struct.pack('<8I', *_IV)
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_MASK = 0xFFFFFFFF
_BUILD_LOCK = threading.Lock()


def _native():
    return _load_native(artifacts_enabled())


@lru_cache(maxsize=2)
def _load_native(allow_build):
    compiler = shutil.which('c++')
    module = Path(__file__)
    source = module.with_name(module.stem + '_native.cpp')
    if compiler is None or os.name != 'posix' or not source.is_file():
        return None
    try:
        with _BUILD_LOCK:
            identity = source.read_bytes() + platform.machine().encode() + platform.system().encode()
            folder = cache_dir() / 'hashing'
            library = folder / (module.stem + '-' + hashlib.sha256(identity).hexdigest()[:16] + '.so')
            if not library.is_file():
                if not allow_build:
                    return None
                folder.mkdir(parents=True, exist_ok=True)
                temporary = library.with_suffix(f'.{os.getpid()}.tmp.so')
                try:
                    subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                                    str(source), '-o', str(temporary)],
                                   check=True, capture_output=True, timeout=60)
                    temporary.replace(library)
                finally:
                    temporary.unlink(missing_ok=True)
            binding = ctypes.CDLL(str(library))
            pointer = ctypes.c_void_p
            binding.cnn3_context_size.argtypes = []
            binding.cnn3_context_size.restype = ctypes.c_size_t
            binding.cnn3_context_align.argtypes = []
            binding.cnn3_context_align.restype = ctypes.c_size_t
            binding.cnn3_init.argtypes = [pointer, pointer, ctypes.c_uint]
            binding.cnn3_init.restype = None
            binding.cnn3_update.argtypes = [pointer, pointer, ctypes.c_size_t]
            binding.cnn3_update.restype = None
            binding.cnn3_finalize.argtypes = [pointer, ctypes.c_uint64, pointer, ctypes.c_size_t]
            binding.cnn3_finalize.restype = None
            binding.cnn3_compress.argtypes = [pointer, pointer, ctypes.c_uint64,
                                              ctypes.c_uint, ctypes.c_uint, pointer]
            binding.cnn3_compress.restype = None
            binding.cnn3_chunk_cv.argtypes = [pointer, ctypes.c_uint, pointer,
                                             ctypes.c_uint64, ctypes.c_uint, pointer]
            binding.cnn3_chunk_cv.restype = None
            binding.cnn3_parent_cv.argtypes = [pointer, pointer, pointer,
                                              ctypes.c_uint, pointer]
            binding.cnn3_parent_cv.restype = None
            if binding.cnn3_context_align() > ctypes.alignment(ctypes.c_uint64):
                return None
            return binding
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None


def _bytes(data):
    if isinstance(data, bytes):
        return data
    view = memoryview(data)
    if not view.c_contiguous:
        raise BufferError('hash input must be C-contiguous')
    return view.tobytes()


def _unsigned(value, label, bits=64):
    value = operator.index(value)
    if not 0 <= value < 1 << bits:
        raise OverflowError(f'{label} must be an unsigned {bits}-bit integer')
    return value


def _rotr(value, count):
    return ((value >> count) | (value << (32 - count))) & _MASK


def _compress(block, cv, counter, block_len, flags):
    words = struct.unpack('<16I', block)
    cv = struct.unpack('<8I', cv)
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
    return struct.pack('<16I', *(state[i] ^ state[i + 8] for i in range(8)),
                       *(state[i + 8] ^ cv[i] for i in range(8)))


class _Output:
    __slots__ = ('block', 'cv', 'counter', 'length', 'flags')

    def __init__(self, block, cv, counter, length, flags):
        self.block, self.cv, self.counter, self.length, self.flags = block, cv, counter, length, flags

    def chaining_value(self):
        return _compress(self.block, self.cv, self.counter, self.length, self.flags)[:32]

    def digest(self, length, seek):
        output = bytearray()
        counter, offset = divmod(seek, 64)
        while len(output) < length:
            block = _compress(self.block, self.cv, counter, self.length, self.flags | 8)
            output.extend(block[offset:offset + length - len(output)])
            offset = 0
            counter += 1
        return bytes(output)


def _chunk_output(data, key, counter, flags):
    cv = key
    for start in range(0, max(len(data), 1), 64):
        block = data[start:start + 64]
        block_flags = flags | (1 if start == 0 else 0)
        if start + 64 >= len(data):
            return _Output(block.ljust(64, b'\0'), cv, counter, len(block), block_flags | 2)
        cv = _compress(block, cv, counter, 64, block_flags)[:32]


def _parent_output(left, right, key, flags):
    return _Output(left + right, key, 0, 64, flags | 4)


class _PythonState:
    __slots__ = ('key', 'flags', 'count', 'chunk', 'stack')

    def __init__(self, key, flags):
        self.key, self.flags = key, flags
        self.count = 0
        self.chunk = bytearray()
        self.stack = []

    def update(self, data):
        data = memoryview(data)
        offset = 0
        while offset < len(data):
            if len(self.chunk) == 1024:
                cv = _chunk_output(bytes(self.chunk), self.key, self.count, self.flags).chaining_value()
                self.count += 1
                total = self.count
                while total & 1 == 0:
                    cv = _parent_output(self.stack.pop(), cv, self.key, self.flags).chaining_value()
                    total >>= 1
                self.stack.append(cv)
                self.chunk.clear()
            take = min(1024 - len(self.chunk), len(data) - offset)
            self.chunk.extend(data[offset:offset + take])
            offset += take

    def digest(self, length, seek):
        output = _chunk_output(bytes(self.chunk), self.key, self.count, self.flags)
        for left in reversed(self.stack):
            output = _parent_output(left, output.chaining_value(), self.key, self.flags)
        return output.digest(length, seek)

    def copy(self):
        other = _PythonState(self.key, self.flags)
        other.count, other.chunk, other.stack = self.count, self.chunk.copy(), self.stack.copy()
        return other


class StreamDigest:
    name = customDecoder(bytes.fromhex('06626c616b6533'))
    digest_size = 32
    block_size = 64
    key_size = 32
    AUTO = -1
    __slots__ = ('_key', '_flags', '_lib', '_state', '_lock')

    def __init__(self, data=None, /, *, key=None, derive_key_context=None,
                 max_threads=1, usedforsecurity=True):
        threads = operator.index(max_threads)
        if threads != self.AUTO and not 1 <= threads <= _MASK:
            raise ValueError('not a valid number of threads')
        if type(usedforsecurity) is not bool:
            raise TypeError('usedforsecurity must be a bool')
        if key is not None and derive_key_context is not None:
            raise ValueError('key and derive_key_context are mutually exclusive')
        key_bytes, flags = _IV_BYTES, 0
        lib = _native()
        if key is not None:
            key_bytes = _bytes(key)
            if len(key_bytes) != self.key_size:
                raise ValueError('key must contain exactly 32 bytes')
            flags = 16
        if derive_key_context is not None:
            if not isinstance(derive_key_context, str):
                raise TypeError('derive_key_context must be a str')
            context = self.__class__.__new__(self.__class__)
            context._initialize(_IV_BYTES, 32, lib)
            context.update(derive_key_context.encode('utf-8'))
            key_bytes, flags = context.digest(), 64
        self._initialize(key_bytes, flags, lib)
        if data is not None:
            self.update(data)

    def _initialize(self, key, flags, lib):
        self._key, self._flags, self._lib = key, flags, lib
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            if self._lib is None:
                self._state = _PythonState(self._key, self._flags)
            else:
                words = (self._lib.cnn3_context_size() + 7) // 8
                self._state = (ctypes.c_uint64 * words)()
                self._lib.cnn3_init(self._state, self._key, self._flags)

    def update(self, data, /):
        data = _bytes(data)
        with self._lock:
            if self._lib is None:
                self._state.update(data)
            elif data:
                self._lib.cnn3_update(self._state, data, len(data))
        return self

    def digest(self, length=32, *, seek=0):
        length, seek = _unsigned(length, 'length'), _unsigned(seek, 'seek')
        if not length:
            return b''
        with self._lock:
            if self._lib is None:
                return self._state.digest(length, seek)
            output = ctypes.create_string_buffer(length)
            self._lib.cnn3_finalize(self._state, seek, output, length)
            return output.raw

    def hexdigest(self, length=32, *, seek=0):
        return self.digest(length, seek=seek).hex()

    def copy(self):
        with self._lock:
            other = self.__class__.__new__(self.__class__)
            other._key, other._flags, other._lib = self._key, self._flags, self._lib
            other._lock = threading.RLock()
            if self._lib is None:
                other._state = self._state.copy()
            else:
                other._state = type(self._state)()
                ctypes.memmove(other._state, self._state, ctypes.sizeof(self._state))
            return other

    def update_mmap(self, path):
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                self.update(block)
        return self
