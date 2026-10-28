from __future__ import annotations
import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import re
from typing import Any
DEFAULT_AGENT = 'LRAEP-trainer/0.1'
MAX_DECODED_PROOF_BYTES = 4 * 1024 * 1024
_HEX = re.compile('[0-9a-fA-F]+\\Z')

class PoolProtocolError(RuntimeError):
    pass

def _integer(value: Any, name: str, minimum: int=0) -> int:
    if type(value) is not int or value < minimum:
        raise PoolProtocolError(f'{name} must be an integer >= {minimum}')
    return value

def _text(value: Any, name: str, maximum: int, *, allow_empty: bool=False) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise PoolProtocolError(f'{name} must be a string of at most {maximum} characters')
    if not value and (not allow_empty) or any((ord(c) < 32 or ord(c) == 127 for c in value)):
        raise PoolProtocolError(f'{name} must contain no control characters and must not be empty')
    return value

def _certificate(value: Any) -> int:
    version = _integer(value, 'certificate version', 1)
    if version not in (1, 2, 3):
        raise PoolProtocolError(f'unsupported PEARL certificate version {version}')
    return version

def _proof_bytes(encoded: str) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise PoolProtocolError('plain_proof must be nonempty canonical base64')
    if len(encoded) > 4 * ((MAX_DECODED_PROOF_BYTES + 2) // 3):
        raise PoolProtocolError('plain_proof exceeds the 4 MiB decoded size limit')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise PoolProtocolError('plain_proof must be nonempty canonical base64') from None
    if not raw or base64.b64encode(raw).decode('ascii') != encoded:
        raise PoolProtocolError('plain_proof must be nonempty canonical base64')
    if len(raw) > MAX_DECODED_PROOF_BYTES:
        raise PoolProtocolError('plain_proof exceeds the 4 MiB decoded size limit')
    return raw

@dataclass(frozen=True)
class PoolJob:
    job_id: str
    header_bytes: bytes
    height: int
    cert_version: int
    target: int
    raw_target: str | None = None

    def __post_init__(self) -> None:
        _text(self.job_id, 'job_id', 128)
        if not isinstance(self.header_bytes, bytes) or len(self.header_bytes) != 76:
            raise PoolProtocolError('header must contain exactly 76 bytes')
        _integer(self.height, 'height')
        _certificate(self.cert_version)
        _integer(self.target, 'target', 1)
        if self.target >= 1 << 256:
            raise PoolProtocolError('target must fit in 256 bits')

class KEXDialect:

    def __init__(self, KEX_uname: str, worker: str, password: str='x', agent: str=DEFAULT_AGENT, request_gzip: bool=False) -> None:
        self.KEX_uname = _text(KEX_uname, 'KEX_uname', 256)
        self.worker = _text(worker, 'worker', 128, allow_empty=True)
        self.password = _text(password, 'password', 1024, allow_empty=True)
        self.agent = _text(agent, 'agent', 128)
        if any((c.isspace() for c in self.KEX_uname + self.worker)):
            raise PoolProtocolError('KEX_uname and worker must contain no whitespace')
        if type(request_gzip) is not bool:
            raise PoolProtocolError('request_gzip must be a boolean')
        self.request_gzip = request_gzip

    @property
    def login(self) -> str:
        return f'{self.KEX_uname}.{self.worker}' if self.worker else self.KEX_uname

    def auth_requests(self, start_id: int=1) -> list[dict[str, Any]]:
        _integer(start_id, 'request ID', 1)
        if self.request_gzip:
            return [{'jsonrpc': '2.0', 'id': start_id, 'method': 'mining.authorize', 'params': {'wallet': self.login, 'agent': self.agent, 'password': self.password, 'type': 'v2'}}]
        return [{'jsonrpc': '2.0', 'id': start_id, 'method': 'mining.subscribe', 'params': [self.agent]}, {'jsonrpc': '2.0', 'id': start_id + 1, 'method': 'mining.authorize', 'params': [self.login, self.password]}]

    @staticmethod
    def accept(response: Mapping[str, Any]) -> bool:
        if not isinstance(response, Mapping) or 'method' in response:
            return False
        return response.get('result') is True and response.get('error') is None

    def negotiate_gzip(self, auth_response: Mapping[str, Any]) -> bool:
        return self.request_gzip and self.accept(auth_response) and (auth_response.get('type') == 'v2')

    def parse_job(self, params: Mapping[str, Any], configured_certificate_version: int | None=None) -> PoolJob:
        if not isinstance(params, Mapping):
            raise PoolProtocolError('mining.notify params must be an object')
        if configured_certificate_version is not None:
            _certificate(configured_certificate_version)
        job_id = _text(params.get('job_id'), 'job_id', 128)
        encoded_header = params.get('header')
        if not isinstance(encoded_header, str) or len(encoded_header) != 152 or _HEX.fullmatch(encoded_header) is None:
            raise PoolProtocolError('header must be exactly 152 hexadecimal characters')
        raw_target = params.get('target')
        if not isinstance(raw_target, str):
            raise PoolProtocolError('target must be a hexadecimal string')
        digits = raw_target[2:] if raw_target.startswith('0x') else raw_target
        if not 1 <= len(digits) <= 64 or _HEX.fullmatch(digits) is None:
            raise PoolProtocolError('target must be a positive 256-bit hexadecimal string')
        target = int(digits, 16)
        version = params.get('cert_version', configured_certificate_version)
        if version is None:
            raise PoolProtocolError('job is missing cert_version; configure it explicitly or update the pool')
        return PoolJob(job_id=job_id, header_bytes=bytes.fromhex(encoded_header), height=_integer(params.get('height'), 'height'), cert_version=_certificate(version), target=target, raw_target=raw_target)

    def submit_request(self, job: PoolJob, plain_proof_b64: str, request_id: int, negotiated_gzip: bool=False) -> dict[str, Any]:
        if not isinstance(job, PoolJob):
            raise PoolProtocolError('submission requires a PoolJob')
        _integer(request_id, 'request ID', 1)
        if type(negotiated_gzip) is not bool or (negotiated_gzip and (not self.request_gzip)):
            raise PoolProtocolError('gzip submissions require an explicitly requested gzip session')
        raw = _proof_bytes(plain_proof_b64)
        encoded = plain_proof_b64
        if negotiated_gzip:
            encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii')
        return {'jsonrpc': '2.0', 'id': request_id, 'method': 'mining.submit', 'params': {'worker': self.login, 'job_id': job.job_id, 'plain_proof': encoded}}
