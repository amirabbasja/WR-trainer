"""JSON configuration with only the S_3 section encrypted."""

import base64
import binascii
import copy
import json
import math
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


FORMAT = "LRAEP-config-text-v1"
DEFAULT_DECRYPT_KEY = "qwe"
MAX_OPEN_TEXT_BYTES = 65536


class ExportError(ValueError):
    pass


def _text_bytes(open_text):
    if not isinstance(open_text, str) or not open_text.strip():
        raise ExportError("decryption key must be nonempty text")
    try:
        value = open_text.encode("utf-8")
    except UnicodeError:
        raise ExportError("decryption key must contain valid UTF-8") from None
    if len(value) > MAX_OPEN_TEXT_BYTES:
        raise ExportError("decryption key exceeds the supported length")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExportError("configuration contains duplicate fields")
        result[key] = value
    return result


def _constant(_):
    raise ExportError("configuration numbers must be finite")


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExportError("configuration numbers must be finite")
    return parsed


def _document(value):
    try:
        result = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant,
                            parse_float=_float)
    except (TypeError, ValueError, RecursionError):
        raise ExportError("configuration must contain valid JSON") from None
    if not isinstance(result, dict):
        raise ExportError("configuration must be an object")
    return result


def _json_bytes(values):
    if not isinstance(values, dict):
        raise ExportError("configuration must be an object")
    try:
        plaintext = json.dumps(values, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError):
        raise ExportError("configuration must contain JSON-compatible values") from None
    _document(plaintext)
    return plaintext


def _cipher(open_text, salt):
    material = _text_bytes(open_text)
    key = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(material)
    return Fernet(base64.urlsafe_b64encode(key))


def _encrypt_object(values, open_text):
    plaintext = _json_bytes(values)
    salt = os.urandom(16)
    ciphertext = _cipher(open_text, salt).encrypt(plaintext).decode("ascii")
    return {"format": FORMAT, "salt": base64.urlsafe_b64encode(salt).decode("ascii"),
            "ciphertext": ciphertext}


def _decrypt_object(envelope, open_text):
    if (not isinstance(envelope, dict) or set(envelope) != {"format", "salt", "ciphertext"}
            or envelope["format"] != FORMAT or not isinstance(envelope["salt"], str)
            or not isinstance(envelope["ciphertext"], str)):
        raise ExportError("unsupported or invalid encrypted configuration format")
    try:
        salt = base64.b64decode(envelope["salt"].encode("ascii"), altchars=b"-_", validate=True)
        if len(salt) != 16 or base64.urlsafe_b64encode(salt).decode("ascii") != envelope["salt"]:
            raise ValueError
    except (ValueError, UnicodeError, binascii.Error):
        raise ExportError("encrypted configuration salt is invalid") from None
    cipher = _cipher(open_text, salt)
    try:
        plaintext = cipher.decrypt(envelope["ciphertext"].encode("ascii"))
    except (InvalidToken, ValueError, UnicodeError):
        raise ExportError("could not decrypt configuration: incorrect key or damaged ciphertext") from None
    return _document(plaintext)


def encode_config_document(values, open_text=DEFAULT_DECRYPT_KEY):
    """Keep ordinary settings readable; encrypt S3 as the on-disk S_3 value."""
    _text_bytes(open_text)
    document = _document(_json_bytes(values))
    if "S_3" in document:
        document = decode_config_document(document, open_text)
    if "S3" in document:
        document["S_3"] = _encrypt_object(document.pop("S3"), open_text)
    return document


def decode_config_document(document, open_text=DEFAULT_DECRYPT_KEY):
    """Return S3 in memory; accept old plain and whole-document exports for migration."""
    if not isinstance(document, dict):
        raise ExportError("configuration must be an object")
    values = copy.deepcopy(document)
    if {"format", "salt", "ciphertext"} & values.keys():
        values = _decrypt_object(values, open_text)
    if "S_3" in values:
        if "S3" in values:
            raise ExportError("configuration must not contain both S3 and S_3")
        values["S3"] = _decrypt_object(values.pop("S_3"), open_text)
    return values
