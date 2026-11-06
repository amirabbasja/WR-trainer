"""Native exact-word transformations and scoped mutable output buffers."""

from contextlib import contextmanager
import ctypes
from functools import lru_cache
from pathlib import Path
import platform


def library_path():
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    if platform.system() != 'Linux' or machine not in ('x86_64', 'aarch64'):
        raise RuntimeError('the native word codec requires a compiled library for this platform')
    return Path(__file__).with_name('bin') / ('linux-' + machine) / 'libword_codec.so'


@lru_cache(maxsize=1)
def _library():
    path = library_path()
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise RuntimeError('native word codec unavailable; build and deploy it from the trainer repository') from error
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    library.wr_codec_abi_version.argtypes = []
    library.wr_codec_abi_version.restype = ctypes.c_uint
    if library.wr_codec_abi_version() != 1:
        raise RuntimeError('unsupported native word codec ABI')
    library.wr_codec_source_id.argtypes = []
    library.wr_codec_source_id.restype = ctypes.c_char_p
    library.wr_codec_transform.argtypes = [pointer, size, ctypes.c_int, pointer, size, ctypes.POINTER(size)]
    library.wr_codec_transform.restype = ctypes.c_int
    library.wr_codec_fixed_decode.argtypes = [pointer, size, pointer, size, ctypes.POINTER(size)]
    library.wr_codec_fixed_decode.restype = ctypes.c_int
    library.wr_codec_wipe.argtypes = [pointer, size]
    library.wr_codec_wipe.restype = None
    return library


def _check(status):
    if status == 4:
        raise ValueError('invalid encoded constant')
    if status:
        raise RuntimeError('native word codec failed with status ' + str(status))


@contextmanager
def _output_buffer(input_bytes, mode):
    library = _library()
    written = ctypes.c_size_t()
    pointer = ctypes.c_char_p(input_bytes)
    arguments = [pointer, len(input_bytes)]
    if mode is None:
        operation = library.wr_codec_fixed_decode
    else:
        operation = library.wr_codec_transform
        arguments.append(mode)
    _check(operation(*arguments, None, 0, ctypes.byref(written)))
    value = bytearray(written.value)
    try:
        output = (ctypes.c_ubyte * len(value)).from_buffer(value)
        try:
            _check(operation(*arguments, output, len(value), ctypes.byref(written)))
            if written.value != len(value):
                raise RuntimeError('native word codec output length changed')
        finally:
            del output
        yield value
    finally:
        wipe = (ctypes.c_ubyte * len(value)).from_buffer(value)
        try:
            library.wr_codec_wipe(wipe, len(value))
        finally:
            del wipe


@contextmanager
def word_buffer(word: str, method: str = 'decode'):
    if not isinstance(word, str):
        raise TypeError('word must be a string')
    if not isinstance(method, str) or method not in ('encode', 'decode'):
        raise ValueError('method must be encode or decode')
    with _output_buffer(word.encode('utf-8', errors='surrogatepass'), int(method == 'encode')) as value:
        yield value


def transform_word(word: str, method: str = 'decode') -> str:
    """Transform an exact table entry; return unknown input unchanged in value."""
    with word_buffer(word, method) as value:
        return value.decode('utf-8', errors='surrogatepass')


def decode_word(word: str) -> str:
    return transform_word(word, 'decode')


def encode_word(word: str) -> str:
    return transform_word(word, 'encode')


def word_matches(placeholder: str, value: object) -> bool:
    """Compare an ASCII protocol value using a cleared mutable native output."""
    if not isinstance(value, str):
        return False
    with word_buffer(placeholder) as expected:
        return len(value) == len(expected) and all(
            ord(character) == byte for character, byte in zip(value, expected)
        )


@contextmanager
def fixed_buffer(encoded: bytes):
    if not isinstance(encoded, bytes):
        raise TypeError('encoded constant must be bytes')
    with _output_buffer(encoded, None) as value:
        yield value
