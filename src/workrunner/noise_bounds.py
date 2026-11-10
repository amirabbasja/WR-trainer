import struct

from .fixed_constants import FixedAlias, constant_buffer


def tensor_domain(is_b=False):
    alias = 'training.tensor.weight_noise_domain' if is_b else 'training.tensor.input_noise_domain'
    return constant_buffer(FixedAlias(alias))


def certificate_salt(digest_type, label):
    with constant_buffer('noise_seed_domain') as domain:
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
