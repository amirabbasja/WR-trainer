from base64 import b64decode


def s5_nonce():
    return b64decode('bWluZXJfbm9uY2U=').decode('ascii')


def s6_noise_seed_prefix():
    return b64decode('cGVhcmwvY2VydC12My9ub2lzZS1zZWVkLw==')


def s7_legacy_alias():
    return b64decode('cGVhcmxoYXNo').decode('ascii')
