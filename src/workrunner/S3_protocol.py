from __future__ import annotations
import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import re
from typing import Any
from . import S3_methods as methods
from . import wire_constants as wire
from .fixed_constants import constant_text

_VERSION_LABEL = b'#KP6n&cV*a#?!)w(Zbc#&dA9'
_BAD_VERSION = b'(#h1)*Vo9^)561t^pS_e!qwBx!Oq0S)53?*!qwEy$jOJ?+5'
_BAD_EVIDENCE = b'*UQGv$@tgR$jHHm%F@)+hsDB&$;ipV%GcB0hs4Io$jQ#c#><Dr#?-=*lK'
_OVERSIZED_EVIDENCE = b'*UQGv$@tgR$jHHm!r#Qg!o$>u)6c?(lZWcg<%h$<#K^<K!-v$)-NJ{<&dScy'
_OVERSIZED_THRESHOLD = b')5g`n!qbP!($v$3!Oqi%&dG<Bl97kS&ePN'
_BAD_ACCOUNT = b'?Bf6U(#gik!iUDm!-vqw)y%@xhsx5_(}%>!$<xNp$%o0vhtSW?)56r(#>Bz'
_BAD_THRESHOLD_TYPE = b')5g`n!qbP!($v$3#lnZihtI;_#>2wI&dSEiht$*6&dI<'
_BAD_THRESHOLD_VALUE = b')5g`n!qbP!($v$3#lnZihu6r|&eP7(!iSZTk&4C6(}&N(-^Rnj#Lmjb%ZJp{)y~Pl'
_NO_JOB_VERSION = b'&B(=v&eVs>&eYV-$-sxi!qwCG(Zbc#&dAA}hs4Oq!Op<a)xw9)(}%*}*UQet&eO}@hsf23($~Yr)53?-&%%fFlm'

def s2():
    return wire.ACCOUNT
_DEFAULT_AGENT = object()


def __getattr__(name):
    if name == 'DEFAULT_AGENT':
        return wire.CLIENT_AGENT
    raise AttributeError(name)


class _Agent:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return wire.CLIENT_AGENT if instance._agent is _DEFAULT_AGENT else instance._agent

    def __set__(self, instance, value):
        instance._agent = value
MAX_DECODED_PROOF_BYTES = 4 * 1024 * 1024
_HEX = re.compile('[0-9a-fA-F]+\\Z')

class S3ProtocolError(RuntimeError):
    pass

def _integer(value: Any, name: str, minimum: int=0) -> int:
    if type(value) is not int or value < minimum:
        raise S3ProtocolError(f'{name} must be an integer >= {minimum}')
    return value

def _text(value: Any, name: str, maximum: int, *, allow_empty: bool=False) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise S3ProtocolError(f'{name} must be a string of at most {maximum} characters')
    if not value and (not allow_empty) or any((ord(c) < 32 or ord(c) == 127 for c in value)):
        raise S3ProtocolError(f'{name} must contain no control characters and must not be empty')
    return value

def _certificate(value: Any) -> int:
    version = _integer(value, constant_text(_VERSION_LABEL), 1)
    if version not in (1, 2, 3):
        raise S3ProtocolError(constant_text(_BAD_VERSION).format(version))
    return version

def _proof_bytes(encoded: str) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise S3ProtocolError(constant_text(_BAD_EVIDENCE))
    if len(encoded) > 4 * ((MAX_DECODED_PROOF_BYTES + 2) // 3):
        raise S3ProtocolError(constant_text(_OVERSIZED_EVIDENCE))
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise S3ProtocolError(constant_text(_BAD_EVIDENCE)) from None
    if not raw or base64.b64encode(raw).decode('ascii') != encoded:
        raise S3ProtocolError(constant_text(_BAD_EVIDENCE))
    if len(raw) > MAX_DECODED_PROOF_BYTES:
        raise S3ProtocolError(constant_text(_OVERSIZED_EVIDENCE))
    return raw

@dataclass(frozen=True)
class S3Job:
    job_id: str
    header_bytes: bytes
    height: int
    cert_version: int
    target: int
    raw_target: str | None = None

    def __post_init__(self) -> None:
        _text(self.job_id, wire.JOB_ID, 128)
        if not isinstance(self.header_bytes, bytes) or len(self.header_bytes) != 76:
            raise S3ProtocolError('header must contain exactly 76 bytes')
        _integer(self.height, 'height')
        _certificate(self.cert_version)
        _integer(self.target, wire.THRESHOLD, 1)
        if self.target >= 1 << 256:
            raise S3ProtocolError(constant_text(_OVERSIZED_THRESHOLD))

class KEXDialect:
    """Build neutral requests; S3Session applies the S3_methods wire mapping."""

    agent = _Agent()

    def __init__(self, KEX_uname: str, worker: str, password: str='x', agent: str=_DEFAULT_AGENT, request_gzip: bool=False) -> None:
        self.KEX_uname = _text(KEX_uname, 'KEX_uname', 256)
        self.worker = _text(worker, wire.DEVICE_ID, 128, allow_empty=True)
        self.password = _text(password, 'password', 1024, allow_empty=True)
        self._agent = _DEFAULT_AGENT if agent is _DEFAULT_AGENT else _text(agent, 'agent', 128)
        if any((c.isspace() for c in self.KEX_uname + self.worker)):
            raise S3ProtocolError(constant_text(_BAD_ACCOUNT))
        if type(request_gzip) is not bool:
            raise S3ProtocolError('request_gzip must be a boolean')
        self.request_gzip = request_gzip

    @property
    def login(self) -> str:
        return f'{self.KEX_uname}.{self.worker}' if self.worker else self.KEX_uname

    def auth_requests(self, start_id: int=1) -> list[dict[str, Any]]:
        _integer(start_id, 'request ID', 1)
        if self.request_gzip:
            return [{'jsonrpc': '2.0', 'id': start_id, 'method': methods.COMPUTE_SESSION, 'params': {s2(): self.login, 'agent': self.agent, 'password': self.password, 'type': 'v2'}}]
        return [{'jsonrpc': '2.0', 'id': start_id, 'method': methods.COMPUTE_REGISTER, 'params': [self.agent]}, {'jsonrpc': '2.0', 'id': start_id + 1, 'method': methods.COMPUTE_SESSION, 'params': [self.login, self.password]}]

    @staticmethod
    def accept(response: Mapping[str, Any]) -> bool:
        if not isinstance(response, Mapping) or 'method' in response:
            return False
        return response.get('result') is True and response.get('error') is None

    def negotiate_gzip(self, auth_response: Mapping[str, Any]) -> bool:
        return self.request_gzip and self.accept(auth_response) and (auth_response.get('type') == 'v2')

    def parse_job(self, params: Mapping[str, Any], configured_certificate_version: int | None=None) -> S3Job:
        if not isinstance(params, Mapping):
            raise S3ProtocolError('work.update params must be an object')
        if configured_certificate_version is not None:
            _certificate(configured_certificate_version)
        job_id = _text(params.get(wire.JOB_ID), wire.JOB_ID, 128)
        encoded_header = params.get('header')
        if not isinstance(encoded_header, str) or len(encoded_header) != 152 or _HEX.fullmatch(encoded_header) is None:
            raise S3ProtocolError('header must be exactly 152 hexadecimal characters')
        raw_target = params.get(wire.THRESHOLD)
        if not isinstance(raw_target, str):
            raise S3ProtocolError(constant_text(_BAD_THRESHOLD_TYPE))
        digits = raw_target[2:] if raw_target.startswith('0x') else raw_target
        if not 1 <= len(digits) <= 64 or _HEX.fullmatch(digits) is None:
            raise S3ProtocolError(constant_text(_BAD_THRESHOLD_VALUE))
        target = int(digits, 16)
        version = params.get(wire.FORMAT_FIELD, configured_certificate_version)
        if version is None:
            raise S3ProtocolError(constant_text(_NO_JOB_VERSION))
        return S3Job(job_id=job_id, header_bytes=bytes.fromhex(encoded_header), height=_integer(params.get('height'), 'height'), cert_version=_certificate(version), target=target, raw_target=raw_target)

    def submission_request(self, job: S3Job, plain_proof_b64: str, request_id: int, negotiated_gzip: bool=False) -> dict[str, Any]:
        if not isinstance(job, S3Job):
            raise S3ProtocolError('submission requires a S3Job')
        _integer(request_id, 'request ID', 1)
        if type(negotiated_gzip) is not bool or (negotiated_gzip and (not self.request_gzip)):
            raise S3ProtocolError('gzip submissions require an explicitly requested gzip session')
        raw = _proof_bytes(plain_proof_b64)
        encoded = plain_proof_b64
        if negotiated_gzip:
            encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii')
        return {'jsonrpc': '2.0', 'id': request_id, 'method': methods.WORK_COMPLETE, 'params': {wire.DEVICE_ID: self.login, wire.JOB_ID: job.job_id, wire.EVIDENCE: encoded}}
