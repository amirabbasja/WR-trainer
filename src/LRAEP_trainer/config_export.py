import base64
import binascii
import json
import math
import os
from pathlib import Path
import re
import tempfile

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


FORMAT = 'LRAEP-config-text-v1'
MAX_OPEN_TEXT_BYTES = 65536


class ExportError(ValueError):
    pass


def _text_bytes(open_text):
    if not isinstance(open_text, str) or not open_text.strip():
        raise ExportError('open_text must be a nonempty text value')
    try:
        value = open_text.encode('utf-8')
    except UnicodeError:
        raise ExportError('open_text must contain valid UTF-8') from None
    if len(value) > MAX_OPEN_TEXT_BYTES:
        raise ExportError('open_text exceeds the supported length')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExportError('configuration contains duplicate fields')
        result[key] = value
    return result


def _constant(_):
    raise ExportError('configuration numbers must be finite')


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExportError('configuration numbers must be finite')
    return parsed


def _document(value):
    try:
        result = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant,
                            parse_float=_float)
    except (TypeError, ValueError, RecursionError):
        raise ExportError('configuration must contain valid JSON') from None
    if not isinstance(result, dict):
        raise ExportError('configuration must be an object')
    return result


def _cipher(open_text, salt):
    material = _text_bytes(open_text)
    key = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(material)
    return Fernet(base64.urlsafe_b64encode(key))


def encode_config_document(values, open_text):
    if not isinstance(values, dict):
        raise ExportError('configuration must be an object')
    _text_bytes(open_text)
    try:
        plaintext = json.dumps(values, ensure_ascii=False, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, RecursionError, UnicodeError):
        raise ExportError('configuration must contain JSON-compatible values') from None
    _document(plaintext)
    salt = os.urandom(16)
    ciphertext = _cipher(open_text, salt).encrypt(plaintext).decode('ascii')
    return {'format': FORMAT, 'salt': base64.urlsafe_b64encode(salt).decode('ascii'),
            'ciphertext': ciphertext}


def decode_config_document(envelope, open_text):
    if (not isinstance(envelope, dict) or set(envelope) != {'format', 'salt', 'ciphertext'}
            or envelope['format'] != FORMAT or not isinstance(envelope['salt'], str)
            or not isinstance(envelope['ciphertext'], str)):
        raise ExportError('unsupported or invalid encoded configuration format')
    try:
        salt = base64.b64decode(envelope['salt'].encode('ascii'), altchars=b'-_', validate=True)
        if len(salt) != 16 or base64.urlsafe_b64encode(salt).decode('ascii') != envelope['salt']:
            raise ValueError
    except (ValueError, UnicodeError, binascii.Error):
        raise ExportError('encoded configuration salt is invalid') from None
    cipher = _cipher(open_text, salt)
    try:
        plaintext = cipher.decrypt(envelope['ciphertext'].encode('ascii'))
    except (InvalidToken, ValueError, UnicodeError):
        raise ExportError('could not decode configuration: incorrect open_text or damaged ciphertext') from None
    return _document(plaintext)


def _env_path(path):
    try:
        return Path(path)
    except (TypeError, ValueError):
        raise ExportError('open_text file path is invalid') from None


def _env_lines(path, *, missing=False):
    try:
        with path.open('r', encoding='utf-8', newline='') as stream:
            return stream.readlines()
    except FileNotFoundError:
        if missing:
            return []
        raise ExportError('could not read the open_text file') from None
    except (OSError, UnicodeError, ValueError):
        raise ExportError('could not read the open_text file') from None


def _env_value(line):
    line = line.rstrip('\r\n').lstrip()
    if not line or line.startswith('#'):
        return None
    if re.match(r'export\s+', line):
        line = re.sub(r'^export\s+', '', line, count=1)
    match = re.match(r'[A-Za-z_][A-Za-z0-9_]*', line)
    if match is None or match.group() != 'open_text':
        return None
    tail = line[match.end():].lstrip()
    if not tail.startswith('='):
        raise ExportError('open_text assignment is malformed')
    value = tail[1:].strip()
    if value.startswith('"'):
        try:
            parsed, end = json.JSONDecoder().raw_decode(value)
        except (TypeError, ValueError, RecursionError):
            raise ExportError('open_text assignment is malformed') from None
        remainder = value[end:].strip()
        if not isinstance(parsed, str) or (remainder and not remainder.startswith('#')):
            raise ExportError('open_text assignment is malformed')
        return parsed
    if value.startswith("'"):
        end = value.find("'", 1)
        if end == -1:
            raise ExportError('open_text assignment is malformed')
        remainder = value[end + 1:].strip()
        if remainder and not remainder.startswith('#'):
            raise ExportError('open_text assignment is malformed')
        return value[1:end]
    return re.split(r'\s+#', value, maxsplit=1)[0].rstrip()


def _find_open_text(lines):
    found = None
    for index, line in enumerate(lines):
        value = _env_value(line)
        if value is None:
            continue
        if found is not None:
            raise ExportError('open_text must appear exactly once in its file')
        found = (index, value)
    return found


def read_open_text(path):
    found = _find_open_text(_env_lines(_env_path(path)))
    if found is None:
        raise ExportError('open_text is missing from its file')
    value = found[1]
    _text_bytes(value)
    return value


def write_open_text(path, text):
    _text_bytes(text)
    path = _env_path(path)
    lines = _env_lines(path, missing=True)
    found = _find_open_text(lines)
    assignment = 'open_text=' + json.dumps(text, ensure_ascii=False)
    if found is not None:
        index = found[0]
        ending = '\r\n' if lines[index].endswith('\r\n') else '\n'
        lines[index] = assignment + ending
    else:
        if lines and not lines[-1].endswith(('\r', '\n')):
            lines[-1] += '\n'
        lines.append(assignment + '\n')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix='.env-', delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, UnicodeError):
        raise ExportError('could not write the open_text file') from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
