from __future__ import annotations
import json
import secrets
COEFFICIENTS = (30, 98, 196, 144, 155, 170, 71, 39, 58, 150, 80, 173, 71, 27, 97, 172, 99, 54, 12, 60, 89, 14, 72, 89, 34, 69, 189, 187, 42, 5, 1, 4, 8, 3, 2, 99, 232, 30, 244, 6, 9, 4, 11, 171, 55, 99, 102, 123)
MAX_FRAME_CHARS = 8 * 1024 * 1024
_HEX = frozenset(b'0123456789abcdefABCDEF')

class CodecError(ValueError):
    pass

def seed_bytes(seed: int | bytes | None=None) -> bytes:
    if seed is None:
        return secrets.token_bytes(4)
    if isinstance(seed, bytes):
        if len(seed) != 4:
            raise CodecError('a byte seed must contain exactly four bytes')
        return seed
    if isinstance(seed, int) and (not isinstance(seed, bool)) and (0 <= seed < 2 ** 32):
        return seed.to_bytes(4, 'big')
    raise CodecError('seed must be four bytes or an unsigned 32-bit integer')

def derive_key(seed: int | bytes) -> bytes:
    number = int.from_bytes(seed_bytes(seed), 'big')
    return bytes((coefficient * number - number % coefficient & 255 for coefficient in COEFFICIENTS))

def _xor(payload: bytes, key: bytes) -> bytes:
    return bytes((value ^ key[index % len(key)] for index, value in enumerate(payload)))

def encode_payload(payload: bytes, seed: int | bytes | None=None) -> bytes:
    if not isinstance(payload, bytes):
        raise CodecError('payload must be bytes')
    selected = seed_bytes(seed)
    if (len(payload) + 4) * 2 > MAX_FRAME_CHARS:
        raise CodecError('frame exceeds the configured byte limit')
    return (selected + _xor(payload, derive_key(selected))).hex().encode('ascii') + b'\n'

def decode_payload(frame: bytes | str) -> bytes:
    if isinstance(frame, str):
        try:
            frame = frame.encode('ascii')
        except UnicodeEncodeError as exc:
            raise CodecError('wire frame must be ASCII hex') from exc
    if not isinstance(frame, bytes):
        raise CodecError('frame must be bytes or ASCII text')
    frame = frame.removesuffix(b'\n').removesuffix(b'\r')
    if len(frame) > MAX_FRAME_CHARS:
        raise CodecError('frame exceeds the configured byte limit')
    if len(frame) <= 8 or len(frame) % 2 or any((value not in _HEX for value in frame)):
        raise CodecError('frame must contain a four-byte seed and an even-length hex payload')
    wire = bytes.fromhex(frame.decode('ascii'))
    return _xor(wire[4:], derive_key(wire[:4]))

def encode_frame(message: dict, seed: int | bytes | None=None) -> bytes:
    if not isinstance(message, dict):
        raise CodecError('protocol messages must be JSON objects')
    try:
        payload = json.dumps(message, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8') + b'\n'
    except (TypeError, ValueError) as exc:
        raise CodecError('message is not a JSON-serializable object') from exc
    return encode_payload(payload, seed)

def decode_frame(frame: bytes | str) -> dict:

    def invalid_constant(value):
        raise CodecError(f'invalid JSON numeric constant: {value}')

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CodecError('decoded JSON contains a duplicate field')
            result[key] = value
        return result
    try:
        message = json.loads(decode_payload(frame), parse_constant=invalid_constant, object_pairs_hook=unique_object)
    except CodecError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise CodecError('decoded payload is not valid UTF-8 JSON') from exc
    if not isinstance(message, dict):
        raise CodecError('protocol messages must be JSON objects')
    return message
