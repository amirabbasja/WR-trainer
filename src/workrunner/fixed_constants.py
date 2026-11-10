from contextlib import contextmanager
from contextvars import ContextVar

from .native_text_codec import FixedAlias as FixedAlias, fixed_buffer, word_buffer


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
def constant_buffer(original: bytes | str):
    buffer = word_buffer(original) if isinstance(original, str) else fixed_buffer(original)
    with buffer as value:
        yield value


def constant_bytes(original: bytes | str) -> bytes:
    with constant_buffer(original) as value:
        return bytes(value)


def constant_text(original: bytes | str) -> str:
    """Resolve a wire name or byte literal into a caller-owned string."""
    if isinstance(original, str) and _neutral_text.get():
        return original
    with constant_buffer(original) as value:
        return value.decode("utf-8")


def constant_ascii_matches(original: bytes | str, value: object) -> bool:
    """Compare ASCII protocol text using a scoped mutable buffer."""
    if not isinstance(value, str):
        return False
    if isinstance(original, str) and _neutral_text.get():
        return original == value
    with constant_buffer(original) as expected:
        return len(value) == len(expected) and all(
            ord(character) == byte for character, byte in zip(value, expected)
        )
