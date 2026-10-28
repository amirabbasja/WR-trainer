from enum import Enum

_CHARSET = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l'
_VALUES = {char: index for index, char in enumerate(_CHARSET)}


class Encoding(Enum):
    BECH32 = 1
    BECH32M = 0x2BC830A3


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


def bech32_encode(hrp, data, encoding=Encoding.BECH32):
    if not isinstance(hrp, str) or not hrp or any(not 33 <= ord(c) <= 126 for c in hrp):
        raise ValueError('Invalid address prefix')
    if hrp.lower() != hrp and hrp.upper() != hrp:
        raise ValueError('Address prefix must use one case')
    values = list(data)
    if any(type(value) is not int or not 0 <= value < 32 for value in values):
        raise ValueError('Address data must use 5-bit values')
    if not isinstance(encoding, Encoding):
        raise ValueError('Invalid address checksum encoding')
    lower = hrp.lower()
    check = _polymod(_expand(lower) + values + [0] * 6) ^ encoding.value
    suffix = ''.join(_CHARSET[value] for value in values)
    suffix += ''.join(_CHARSET[(check >> (5 * (5 - i))) & 31] for i in range(6))
    result = lower + '1' + suffix
    if len(result) > 90:
        raise ValueError('Address exceeds 90 characters')
    return result.upper() if hrp.isupper() else result


def bech32_decode(address):
    if not isinstance(address, str) or not 8 <= len(address) <= 90:
        return None, None, None
    if any(not 33 <= ord(c) <= 126 for c in address):
        return None, None, None
    if address.lower() != address and address.upper() != address:
        return None, None, None
    address = address.lower()
    separator = address.rfind('1')
    if separator < 1 or separator + 7 > len(address):
        return None, None, None
    try:
        values = [_VALUES[c] for c in address[separator + 1:]]
    except KeyError:
        return None, None, None
    check = _polymod(_expand(address[:separator]) + values)
    try:
        encoding = Encoding(check)
    except ValueError:
        return None, None, None
    return address[:separator], values[:-6], encoding


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
