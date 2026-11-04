from base64 import b85decode
from contextlib import contextmanager


@contextmanager
def constant_buffer(encoded: bytes):
    value = bytearray(b85decode(encoded))
    try:
        for index in range(len(value)):
            value[index] ^= 0xA7
        yield value
    finally:
        for index in range(len(value)):
            value[index] = 0


def constant_bytes(encoded: bytes) -> bytes:
    with constant_buffer(encoded) as value:
        return bytes(value)


def constant_text(encoded: bytes) -> str:
    with constant_buffer(encoded) as value:
        return value.decode("utf-8")
