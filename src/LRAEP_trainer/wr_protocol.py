from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any
from .pearl_requests import build_submit_request
from .pool_protocol import DEFAULT_AGENT, PoolJob, PoolProtocolError, _text
_HEX = re.compile('[0-9a-fA-F]+\\Z')
HEADER_MASK = bytes.fromhex('9d2f73c411a85be0367ad14e920cbf65f32884da176eb941cc530fa67de21974')
WORK_PREFIX_MASK = bytes.fromhex('4dc0d47dff723cf8f75a583b596cf848f5106dd0d0aa7bdded9d4d566a929ffaabea8985cf34a26af1b118e2c697f283f0caadd643b0975390a17c79aae32e69cf166b57a6f64dd02914e7e4')

class UnknownWRRequest(PoolProtocolError):
    pass

def _hex_bytes(value: Any, length: int, label: str) -> bytes:
    if not isinstance(value, str) or len(value) != length * 2 or _HEX.fullmatch(value) is None:
        raise PoolProtocolError(f'PearlHash {label} must contain exactly {length} hexadecimal bytes')
    return bytes.fromhex(value)

def decode_notify_payload(value: str) -> bytes:
    payload = _hex_bytes(value, 200, 'notify payload')
    header = bytes((byte ^ WORK_PREFIX_MASK[index] ^ HEADER_MASK[index % 32] for index, byte in enumerate(payload[:76])))
    nbits = int.from_bytes(header[72:76], 'little')
    exponent, mantissa = (nbits >> 24, nbits & 8388607)
    target = mantissa >> 8 * (3 - exponent) if exponent <= 3 else mantissa << 8 * (exponent - 3)
    if nbits & 8388608 or not 0 < target < 1 << 256:
        raise PoolProtocolError('PearlHash decoded header has an invalid compact target')
    return header

@dataclass(frozen=True)
class WRJob(PoolJob):
    notify_payload: bytes = b''
    worker_nonce: bytes | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cert_version != 3 or self.height != 0:
            raise PoolProtocolError('Only observed PearlHash generation 3 is supported')
        if not isinstance(self.notify_payload, bytes) or len(self.notify_payload) != 200:
            raise PoolProtocolError('PearlHash notify payload must contain 200 bytes')
        if self.worker_nonce is not None and (not isinstance(self.worker_nonce, bytes) or len(self.worker_nonce) != 8):
            raise PoolProtocolError('PearlHash worker nonce must contain eight bytes')

@dataclass(frozen=True)
class WRShare:
    miner_nonce: bytes
    worker_nonce: bytes
    m: int
    n: int
    k: int
    rank: int
    tile_h: int
    tile_w: int
    a_rows: tuple[int, ...]
    b_cols: tuple[int, ...]
    sb: int = 1

    def request(self, request_id: int) -> dict[str, Any]:
        try:
            return build_submit_request(request_id, miner_nonce=self.miner_nonce, worker_nonce=self.worker_nonce, m=self.m, n=self.n, k=self.k, rank=self.rank, tile_h=self.tile_h, tile_w=self.tile_w, sb=self.sb, a_rows=self.a_rows, b_cols=self.b_cols)
        except (ValueError, TypeError) as error:
            raise PoolProtocolError('Invalid PearlHash share fields') from error

class WRDialect:

    def __init__(self, WR_wallet: str, worker: str, agent: str=DEFAULT_AGENT) -> None:
        self.WR_wallet = _text(WR_wallet, 'WR_wallet', 256)
        self.worker = _text(worker, 'worker', 128, allow_empty=True)
        self.agent = _text(agent, 'agent', 128)
        if any((character.isspace() for character in WR_wallet + worker)):
            raise PoolProtocolError('WR_wallet and worker must contain no whitespace')

    def auth_requests(self, start_id: int=1) -> list[dict[str, Any]]:
        if type(start_id) is not int or not 0 <= start_id < 1 << 64:
            raise PoolProtocolError('PearlHash request ID must be an unsigned 64-bit integer')
        return [{'id': start_id, 'method': 'login', 'params': [self.WR_wallet, self.worker, self.agent]}]

    @staticmethod
    def report_request(gpu_name: str, hashrate: float, request_id: int) -> dict[str, Any]:
        if type(request_id) is not int or not 0 <= request_id < 1 << 64:
            raise PoolProtocolError('PearlHash request ID must be an unsigned 64-bit integer')
        name = _text(gpu_name, 'gpu_name', 256)
        if not name.strip() or not name.isprintable():
            raise PoolProtocolError('gpu_name must contain printable GPU identification')
        if isinstance(hashrate, bool) or not isinstance(hashrate, (int, float)):
            raise PoolProtocolError('hashrate must be a nonnegative finite number in H/s')
        try:
            rate = float(hashrate)
        except OverflowError:
            raise PoolProtocolError('hashrate must be a nonnegative finite number in H/s') from None
        if not math.isfinite(rate) or rate < 0:
            raise PoolProtocolError('hashrate must be a nonnegative finite number in H/s')
        return {'id': request_id, 'method': 'report_info', 'params': [{'hashrate': rate, 'name': name}]}

    @staticmethod
    def accept(response: Mapping[str, Any]) -> bool:
        return isinstance(response, Mapping) and 'method' not in response and (type(response.get('status')) is int) and (response['status'] == 1) and (response.get('result') is True) and (response.get('error') is None)

    @staticmethod
    def negotiate_gzip(auth_response: Mapping[str, Any]) -> bool:
        return False

    @staticmethod
    def classify(response: Mapping[str, Any]) -> str:
        if isinstance(response, Mapping) and isinstance(response.get('method'), str):
            raise UnknownWRRequest('Unknown PearlHash server method')
        if not isinstance(response, Mapping) or set(response) - {'id', 'status', 'result', 'error'}:
            raise PoolProtocolError('Unknown PearlHash server message fields or method')
        request_id, status = (response.get('id'), response.get('status'))
        if type(request_id) is not int or not -1 <= request_id < 1 << 64 or type(status) is not int or (status not in (0, 1)):
            raise PoolProtocolError('Unknown PearlHash response envelope')
        result = response.get('result')
        if request_id >= 0 and type(result) is bool:
            return 'reply'
        if request_id != -1 or status != 1 or response.get('error') is not None or (not isinstance(result, list)) or (not result):
            raise PoolProtocolError('Unknown PearlHash server request or response shape')
        if result[0] == 'extranonce':
            if len(result) != 2:
                raise PoolProtocolError('Unsupported PearlHash extranonce schema')
            _hex_bytes(result[1], 8, 'extranonce')
            return 'extranonce'
        if result[0] == 'notify':
            if len(result) != 5 or not all((isinstance(field, str) for field in result)):
                raise PoolProtocolError('Unsupported PearlHash notify schema')
            return 'notify'
        if isinstance(result[0], str):
            raise UnknownWRRequest('Unknown PearlHash server notification')
        raise PoolProtocolError('Malformed PearlHash notification name')

    @staticmethod
    def parse_extranonce(result: list) -> bytes:
        if not isinstance(result, list) or len(result) != 2 or result[0] != 'extranonce':
            raise PoolProtocolError('Unsupported PearlHash extranonce schema')
        return _hex_bytes(result[1], 8, 'extranonce')

    @staticmethod
    def parse_job(result: list, configured_certificate_version: int | None=None, worker_nonce: bytes | None=None) -> WRJob:
        if not isinstance(result, list) or len(result) != 5 or result[0] != 'notify' or (not all((isinstance(field, str) for field in result))):
            raise PoolProtocolError('Unsupported PearlHash notify schema')
        if result[4] != '3':
            raise PoolProtocolError('Only observed PearlHash generation 3 is supported')
        if configured_certificate_version is not None and (type(configured_certificate_version) is not int or configured_certificate_version != 3):
            raise PoolProtocolError('PearlHash generation 3 requires certificate version 3')
        payload = _hex_bytes(result[1], 200, 'notify payload')
        target_bytes = _hex_bytes(result[2], 32, 'share target')
        return WRJob(job_id=_text(result[3], 'job_id', 128), header_bytes=decode_notify_payload(result[1]), height=0, cert_version=3, target=int.from_bytes(target_bytes, 'little'), raw_target=result[2], notify_payload=payload, worker_nonce=worker_nonce)

    @staticmethod
    def submit_request(job: WRJob, share: WRShare, request_id: int, negotiated_gzip: bool=False) -> dict[str, Any]:
        if not isinstance(job, WRJob) or not isinstance(share, WRShare):
            raise PoolProtocolError('PearlHash submission requires its job and compact share')
        if negotiated_gzip:
            raise PoolProtocolError('PearlHash does not support KEX gzip proof submissions')
        if job.worker_nonce is None or share.worker_nonce != job.worker_nonce:
            raise PoolProtocolError('PearlHash share worker nonce does not match its job')
        return share.request(request_id)
