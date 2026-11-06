from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any
from .S6_requests import build_submission_request
from .S3_protocol import S3Job, S3ProtocolError, _text, _Agent, _DEFAULT_AGENT
from . import wire_constants as wire
from .fixed_constants import constant_buffer, constant_bytes, constant_text

_BAD_COMPACT_THRESHOLD = b'^pJ<c!o<kK!o!Ep!p6hG)rZf<)Q85&htA2-#>>vbhs4Or*T%%thttN@z{1l'
_BAD_DEVICE_SEED = b'^pJ<p$koik)rZN*$;85k%F@)+hs4Oq)5gxphr-Ul&(nv+-qXU='
_BAD_RESULT = b'?#a={%g)1x^pJ<t&&Jilhr!Om%fr+'
_BAD_ACCOUNT = b'@b&ogm50X3!-vqw)y%@xhsx5_(}%>!$<xNp$%o0vhtSW?)56r(#>Bz'
_BAD_RATE = b'&&Jfx)yC7phsx5_(}%^vhsKA=$jQmVz{b<g(ZYwp&dJWx!iUMy%EiLfhtA1|?}*d'
_UPDATE_PAYLOAD = b'$;i{r!QO}0#@@@w#=`'
_UPDATE_SIZE_ERROR = b'^pJ<i$kWcj-iO!5-pk0w!-vY!)YFH=$jQ^j&dG<BmzRgd-qXU='
_UPDATE_SCHEMA_ERROR = b'^2yZF*Vo9^)561t^pJ<i$kWcj-iOr0&%(;a'
_BAD_SERVER_SEED = b'^2yZF*Vo9^)561t^pJ<b-_zB`$;ipX!iUtv&%(;a'
_BAD_VERSION = b'^pJ<Z!pXwb#?#Ko$%mAO)xy@&&eg)yhs46w)6T)p#KzOYhta~-)XvDshm-'
_RESULT_THRESHOLD = b')X&D%!iUqw)xg5j'
_MISSING_RESULT = b'^pJ<t(#6Wo)YQ(%$%oa#*3!<^!qkV()6|E}$i;`o$-{@l$jaBo#M6h=&&Jil'
_BAD_COMPRESSION = b'^pJ<c$imcz$;i`()Y8}2$ko$_?Bf52z}?Q*hu77}$iauy(#6Wo)YQ(%$<z'
_SEED_MISMATCH = b'^pJ<t&&JilhtSB?%)-@&$;ipX!iU4i!qkV!$kT_)#?!>lhtAW~ht0^v'

_HEADER_MASK = b'I*8O`wh#PAk==IbHLDoHRF5Nlu*n|gYxJlA+C{$8'

_WORK_PREFIX_MASK = b'>SuG>Sk;?fQ2qa${>xwQQn$)?cMaTnN;>NC%{4e(3{5T~Xp;raRu;cSVK7xBS8WP$<QFjXHwN3@4Md5~XtB)j0a5CAj<Y~R'

_HEX = re.compile('[0-9a-fA-F]+\\Z')


def header_mask():
    return constant_bytes(_HEADER_MASK)


def work_prefix_mask():
    return constant_bytes(_WORK_PREFIX_MASK)

class UnknownWRRequest(S3ProtocolError):
    pass

def _hex_bytes(value: Any, length: int, label: str) -> bytes:
    if not isinstance(value, str) or len(value) != length * 2 or _HEX.fullmatch(value) is None:
        raise S3ProtocolError(f"{constant_text(b'^pJ-')}{label} must contain exactly {length} hexadecimal bytes")
    return bytes.fromhex(value)

def decode_update_payload(value: str) -> bytes:
    payload = _hex_bytes(value, 200, constant_text(_UPDATE_PAYLOAD))
    with constant_buffer(_WORK_PREFIX_MASK) as prefix, constant_buffer(_HEADER_MASK) as mask:
        header = bytes((byte ^ prefix[index] ^ mask[index % 32] for index, byte in enumerate(payload[:76])))
    compact_limit = int.from_bytes(header[72:76], 'little')
    exponent, mantissa = (compact_limit >> 24, compact_limit & 8388607)
    target = mantissa >> 8 * (3 - exponent) if exponent <= 3 else mantissa << 8 * (exponent - 3)
    if compact_limit & 8388608 or not 0 < target < 1 << 256:
        raise S3ProtocolError(constant_text(_BAD_COMPACT_THRESHOLD))
    return header

@dataclass(frozen=True)
class WRJob(S3Job):
    update_payload: bytes = b''
    device_seed: bytes | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cert_version != 3 or self.height != 0:
            raise S3ProtocolError(constant_text(b'=*i38hsed$!qw5j!-w>chrq(g!qvvp&dAA!l!wmLht$&7*T~h=!ov'))
        if not isinstance(self.update_payload, bytes) or len(self.update_payload) != 200:
            raise S3ProtocolError(constant_text(_UPDATE_SIZE_ERROR))
        if self.device_seed is not None and (not isinstance(self.device_seed, bytes) or len(self.device_seed) != 8):
            raise S3ProtocolError(constant_text(_BAD_DEVICE_SEED))

@dataclass(frozen=True)
class WRSubmission:
    local_seed: bytes
    device_seed: bytes
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
            return build_submission_request(request_id, local_seed=self.local_seed, device_seed=self.device_seed, m=self.m, n=self.n, k=self.k, rank=self.rank, tile_h=self.tile_h, tile_w=self.tile_w, sb=self.sb, a_rows=self.a_rows, b_cols=self.b_cols)
        except (ValueError, TypeError) as error:
            raise S3ProtocolError(constant_text(_BAD_RESULT)) from error

class WRDialect:
    agent = _Agent()

    def __init__(self, WR_S2: str, worker: str, agent: str=_DEFAULT_AGENT) -> None:
        self.WR_S2 = _text(WR_S2, 'WR_S2', 256)
        self.worker = _text(worker, wire.DEVICE_ID, 128, allow_empty=True)
        self._agent = _DEFAULT_AGENT if agent is _DEFAULT_AGENT else _text(agent, 'agent', 128)
        if any((character.isspace() for character in WR_S2 + worker)):
            raise S3ProtocolError(constant_text(_BAD_ACCOUNT))

    def auth_requests(self, start_id: int=1) -> list[dict[str, Any]]:
        if type(start_id) is not int or not 0 <= start_id < 1 << 64:
            raise S3ProtocolError(constant_text(b'^pJ<u!q(El)YFIV<A=)9)YFH>!iUDmhtkQ^&cMmS!-tWRip9>;htA2<!ob4S'))
        return [{'id': start_id, 'method': wire.LOGIN, 'params': [self.WR_S2, self.worker, self.agent]}]

    @staticmethod
    def report_request(gpu_name: str, work_rate: float, request_id: int) -> dict[str, Any]:
        if type(request_id) is not int or not 0 <= request_id < 1 << 64:
            raise S3ProtocolError(constant_text(b'^pJ<u!q(El)YFIV<A=)9)YFH>!iUDmhtkQ^&cMmS!-tWRip9>;htA2<!ob4S'))
        name = _text(gpu_name, 'gpu_name', 256)
        if not name.strip() or not name.isprintable():
            raise S3ProtocolError('gpu_name must contain printable GPU identification')
        if isinstance(work_rate, bool) or not isinstance(work_rate, (int, float)):
            raise S3ProtocolError(constant_text(_BAD_RATE))
        try:
            rate = float(work_rate)
        except OverflowError:
            raise S3ProtocolError(constant_text(_BAD_RATE)) from None
        if not math.isfinite(rate) or rate < 0:
            raise S3ProtocolError(constant_text(_BAD_RATE))
        return {'id': request_id, 'method': wire.DEVICE_REPORT, 'params': [{wire.RATE: rate, 'name': name}]}

    @staticmethod
    def accept(response: Mapping[str, Any]) -> bool:
        return isinstance(response, Mapping) and 'method' not in response and (type(response.get('status')) is int) and (response['status'] == 1) and (response.get('result') is True) and (response.get('error') is None)

    @staticmethod
    def negotiate_gzip(auth_response: Mapping[str, Any]) -> bool:
        return False

    @staticmethod
    def classify(response: Mapping[str, Any]) -> str:
        if isinstance(response, Mapping) and isinstance(response.get('method'), str):
            raise UnknownWRRequest(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZQ$)6dAm'))
        if not isinstance(response, Mapping) or set(response) - {'id', 'status', 'result', 'error'}:
            raise S3ProtocolError(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZQ$)YQhn!iT}m!pp<dhsf23%EHsn$in'))
        request_id, status = (response.get('id'), response.get('status'))
        if type(request_id) is not int or not -1 <= request_id < 1 << 64 or type(status) is not int or (status not in (0, 1)):
            raise S3ProtocolError(constant_text(b'^2yA}$k54$^pJ<u!qnHu$<)G!!pYIX%gEQl'))
        result = response.get('result')
        if request_id >= 0 and type(result) is bool:
            return 'reply'
        if request_id != -1 or status != 1 or response.get('error') is not None or (not isinstance(result, list)) or (not result):
            raise S3ProtocolError(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZx>*3!b%(}&2_ht<N=*T~7#!iUt)#@E6'))
        if wire.text_matches('SERVER_SEED', result[0]):
            if len(result) != 2:
                raise S3ProtocolError(constant_text(_BAD_SERVER_SEED))
            _hex_bytes(result[1], 8, wire.SERVER_SEED)
            return result[0]
        if result[0] == wire.JOB_UPDATE:
            if len(result) != 5 or not all((isinstance(field, str) for field in result)):
                raise S3ProtocolError(constant_text(_UPDATE_SCHEMA_ERROR))
            return wire.JOB_UPDATE
        if isinstance(result[0], str):
            raise UnknownWRRequest(constant_text(b'^2yA}$k54$^pJ<t!qw5j)rZN*)6T)p#KzOk$jJ'))
        raise S3ProtocolError(constant_text(b'>c-2#$koci!-w>chsns(&cV*a#?#Ko$%o0t%EA'))

    @staticmethod
    def parse_server_seed(result: list) -> bytes:
        if not isinstance(result, list) or len(result) != 2 or not wire.text_matches('SERVER_SEED', result[0]):
            raise S3ProtocolError(constant_text(_BAD_SERVER_SEED))
        return _hex_bytes(result[1], 8, wire.SERVER_SEED)

    @staticmethod
    def parse_job(result: list, configured_certificate_version: int | None=None, device_seed: bytes | None=None) -> WRJob:
        if not isinstance(result, list) or len(result) != 5 or result[0] != wire.JOB_UPDATE or (not all((isinstance(field, str) for field in result))):
            raise S3ProtocolError(constant_text(_UPDATE_SCHEMA_ERROR))
        if result[4] != '3':
            raise S3ProtocolError(constant_text(b'=*i38hsed$!qw5j!-w>chrq(g!qvvp&dAA!l!wmLht$&7*T~h=!ov'))
        if configured_certificate_version is not None and (type(configured_certificate_version) is not int or configured_certificate_version != 3):
            raise S3ProtocolError(constant_text(_BAD_VERSION))
        payload = _hex_bytes(result[1], 200, constant_text(_UPDATE_PAYLOAD))
        target_bytes = _hex_bytes(result[2], 32, constant_text(_RESULT_THRESHOLD))
        return WRJob(job_id=_text(result[3], wire.JOB_ID, 128), header_bytes=decode_update_payload(result[1]), height=0, cert_version=3, target=int.from_bytes(target_bytes, 'little'), raw_target=result[2], update_payload=payload, device_seed=device_seed)

    @staticmethod
    def submission_request(job: WRJob, submission: WRSubmission, request_id: int, negotiated_gzip: bool=False) -> dict[str, Any]:
        if not isinstance(job, WRJob) or not isinstance(submission, WRSubmission):
            raise S3ProtocolError(constant_text(_MISSING_RESULT))
        if negotiated_gzip:
            raise S3ProtocolError(constant_text(_BAD_COMPRESSION))
        if job.device_seed is None or submission.device_seed != job.device_seed:
            raise S3ProtocolError(constant_text(_SEED_MISMATCH))
        return submission.request(request_id)
