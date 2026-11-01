from base64 import b64decode

S5_NONCE = b64decode('bWluZXJfbm9uY2U=').decode('ascii')
S6_NOISE_SEED_PREFIX = b64decode('cGVhcmwvY2VydC12My9ub2lzZS1zZWVkLw==')
S7_LEGACY_ALIAS = b64decode('cGVhcmxoYXNo').decode('ascii')
