import ctypes
import struct

from .CNN_3ekalb import _bytes, _compress, _native, _unsigned


def compress(block, cv, counter, block_len, flags):
    if not isinstance(block, bytes):
        block = struct.pack('<16I', *block)
    if not isinstance(cv, bytes):
        cv = struct.pack('<8I', *cv)
    if len(block) != 64 or len(cv) != 32:
        raise ValueError('compression requires a 64-byte block and 32-byte chaining value')
    counter = _unsigned(counter, 'counter')
    block_len = _unsigned(block_len, 'block length', 32)
    flags = _unsigned(flags, 'flags', 32)
    if block_len > 64:
        raise ValueError('block length must be at most 64')
    native = _native()
    if native is None:
        return _compress(block, cv, counter, block_len, flags)
    output = ctypes.create_string_buffer(64)
    native.cnn3_compress(block, cv, counter, block_len, flags, output)
    return output.raw


def chunk_cv(data, key, index=0, root=False):
    key = _bytes(key)
    if not isinstance(data, bytes) or len(data) > 1024 or len(key) != 32 or type(index) is not int or not 0 <= index < 1 << 64:
        raise ValueError('invalid keyed chunk')
    native = _native()
    if native is not None:
        output = ctypes.create_string_buffer(32)
        native.cnn3_chunk_cv(data, len(data), key, index, bool(root), output)
        return output.raw
    cv = key
    blocks = [data[i:i + 64] for i in range(0, len(data), 64)] or [b'']
    for i, block in enumerate(blocks):
        last = i == len(blocks) - 1
        flags = 16 | (1 if i == 0 else 0) | (2 if last else 0) | (8 if last and root else 0)
        cv = compress(block.ljust(64, b'\0'), cv, index, len(block), flags)[:32]
    return cv


def parent_cv(left, right, key, root=False):
    left, right, key = _bytes(left), _bytes(right), _bytes(key)
    if len(left) != 32 or len(right) != 32 or len(key) != 32:
        raise ValueError('invalid keyed parent')
    native = _native()
    if native is not None:
        output = ctypes.create_string_buffer(32)
        native.cnn3_parent_cv(left, right, key, bool(root), output)
        return output.raw
    return compress(left + right, key, 0, 64, 16 | 4 | (8 if root else 0))[:32]
