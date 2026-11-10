"""Decode Bech32 and Bech32m identities used by the WR gateway."""

from enum import Enum
from . import training_constants as wire
from .fixed_constants import FixedAlias, constant_text

_CHARSET = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l'
_VALUES = {char: index for index, char in enumerate(_CHARSET)}
_BAD_DEVICE_TYPE = FixedAlias('training.validation.worker_identity_type')
_BAD_DEVICE_ID = FixedAlias('training.validation.coordinator_identity_whitespace_template')


def validate_login_identity(settings, protocol, *, validate_identity=True):
    identity_field = 'WR_S2' if protocol == 'WR' else 'KEX_uname'
    identity = settings.get(identity_field, '')
    worker = settings.get(wire.DEVICE_ID, '')
    if not isinstance(identity, str) or not identity.strip():
        description = 'training identity' if protocol == 'WR' else 'session name or training identity'
        raise ValueError(f"{constant_text(FixedAlias('training.validation.coordinator_config_prefix'))}{identity_field} in your configuration with your {description}")
    if not isinstance(worker, str):
        raise ValueError(constant_text(_BAD_DEVICE_TYPE))
    if any(character.isspace() for character in identity + worker):
        raise ValueError(constant_text(_BAD_DEVICE_ID).format(identity_field))
    if wire.has_identity_mode_prefix(identity):
        raise ValueError('remove the unsupported identity mode prefix to use S3 training')
    if validate_identity and identity.startswith(wire.IDENTITY_PREFIX):
        from job_gateway.record_format.training_assembly import identity_to_output_payload
        try:
            identity_to_output_payload(identity)
        except ValueError:
            raise ValueError(f'S3.{identity_field} must contain a valid S6 Version-1 training identity') from None


class Encoding(Enum):
    STANDARD = 1  # Bech32
    EXTENDED = 0x2BC830A3  # Bech32m


def _polymod(values):
    checksum = 1
    generators = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    for value in values:
        top = checksum >> 25
        checksum = ((checksum & 0x1FFFFFF) << 5) ^ value
        for bit, generator in enumerate(generators):
            if (top >> bit) & 1:
                checksum ^= generator
    return checksum


def _expand(hrp):
    return [ord(char) >> 5 for char in hrp] + [0] + [ord(char) & 31 for char in hrp]


def decode_training_identity(identity):
    """Return the prefix, 5-bit payload, and checksum format."""
    if not isinstance(identity, str) or not 8 <= len(identity) <= 90:
        return None, None, None
    if any(not 33 <= ord(c) <= 126 for c in identity):
        return None, None, None
    if identity.lower() != identity and identity.upper() != identity:
        return None, None, None
    identity = identity.lower()
    separator = identity.rfind('1')
    if separator < 1 or separator + 7 > len(identity):
        return None, None, None
    try:
        values = [_VALUES[c] for c in identity[separator + 1:]]
    except KeyError:
        return None, None, None
    check = _polymod(_expand(identity[:separator]) + values)
    try:
        encoding = Encoding(check)
    except ValueError:
        return None, None, None
    return identity[:separator], values[:-6], encoding


def convertbits(data, frombits, tobits, pad=True):
    if type(frombits) is not int or type(tobits) is not int or not 1 <= frombits <= 8 or not 1 <= tobits <= 8:
        raise ValueError('Bit widths must be between 1 and 8')
    accumulator = bits = 0
    result = []
    mask = (1 << tobits) - 1
    max_accumulator = (1 << (frombits + tobits - 1)) - 1
    for value in data:
        if type(value) is not int or not 0 <= value < 1 << frombits:
            return None
        accumulator = ((accumulator << frombits) | value) & max_accumulator
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            result.append((accumulator >> bits) & mask)
    if pad:
        if bits:
            result.append((accumulator << (tobits - bits)) & mask)
    elif bits >= frombits or ((accumulator << (tobits - bits)) & mask):
        return None
    return result
