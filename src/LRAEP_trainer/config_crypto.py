import os
from cryptography.fernet import Fernet, InvalidToken
KEY_ENV = 'LRAEP_CONFIG_KEY'

class EncryptionError(ValueError):
    pass

def generate_key() -> str:
    return Fernet.generate_key().decode('ascii')

def _cipher(key=None) -> Fernet:
    if key is None:
        key = os.environ.get(KEY_ENV)
    if key is None:
        raise EncryptionError(f'supply an encryption key or set {KEY_ENV}')
    try:
        if isinstance(key, str):
            key = key.encode('ascii')
        if not isinstance(key, bytes):
            raise ValueError
        return Fernet(key)
    except (TypeError, ValueError, UnicodeError):
        raise EncryptionError('encryption key must be a valid Fernet key') from None

def validate_key(key):
    _cipher(key)
    return key

def encode_string(value: str, key=None) -> str:
    if not isinstance(value, str):
        raise EncryptionError('value to encrypt must be a string')
    cipher = _cipher(key)
    try:
        return cipher.encrypt(value.encode('utf-8')).decode('ascii')
    except UnicodeError:
        raise EncryptionError('value to encrypt must contain valid UTF-8') from None

def decode_string(value: str, key=None) -> str:
    if not isinstance(value, str):
        raise EncryptionError('value to decrypt must be a string')
    cipher = _cipher(key)
    try:
        return cipher.decrypt(value.encode('ascii')).decode('utf-8')
    except (InvalidToken, ValueError, UnicodeError):
        raise EncryptionError('could not decrypt data: incorrect key or damaged ciphertext') from None
