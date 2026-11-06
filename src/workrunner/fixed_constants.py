from contextlib import contextmanager
from contextvars import ContextVar

from .word_codec import fixed_buffer, word_buffer


_neutral_text = ContextVar('neutral_protocol_text', default=False)


@contextmanager
def neutral_protocol_text():
    """Keep mapped protocol names neutral until the native network boundary."""
    token = _neutral_text.set(True)
    try:
        yield
    finally:
        _neutral_text.reset(token)


def using_neutral_protocol_text():
    return _neutral_text.get()


@contextmanager
def constant_buffer(encoded: bytes | str):
    buffer = word_buffer(encoded) if isinstance(encoded, str) else fixed_buffer(encoded)
    with buffer as value:
        yield value


def constant_bytes(encoded: bytes | str) -> bytes:
    with constant_buffer(encoded) as value:
        return bytes(value)


def constant_text(encoded: bytes | str) -> str:
    """Decode on demand; the caller owns the returned immutable string."""
    if isinstance(encoded, str) and _neutral_text.get():
        return encoded
    with constant_buffer(encoded) as value:
        return value.decode("utf-8")


def constant_ascii_matches(encoded: bytes | str, value: object) -> bool:
    """Compare ASCII protocol text using a cleared native-decoded buffer."""
    if not isinstance(value, str):
        return False
    if isinstance(encoded, str) and _neutral_text.get():
        return encoded == value
    with constant_buffer(encoded) as expected:
        return len(value) == len(expected) and all(
            ord(character) == byte for character, byte in zip(value, expected)
        )
