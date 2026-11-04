import struct

from .fixed_constants import constant_buffer


def tensor_domain(is_b=False):
    return constant_buffer(b'<@nRW$<)Zzr>Cc<r>Cc<r>Cc<r>Cc<r>Cc<r>Cc<' if is_b else b'=J?aX$<)Zzr>Cc<r>Cc<r>Cc<r>Cc<r>Cc<r>Cc<')


def certificate_salt(digest_type, label):
    with constant_buffer(b'*TTlt%ZS9n)zga6l!(d5&eXz+)WX8Uhy') as domain:
        domain.extend(label)
        return digest_type(domain).digest()


def noise_draw(digest_type, index, label, key, slot):
    message = bytearray(64)
    try:
        struct.pack_into('<I', message, slot * 4, index + 1)
        message[32:] = label
        return digest_type(message, key=key).digest()
    finally:
        for position in range(len(message)):
            message[position] = 0
