from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any
from .S6_requests import build_submission_request
from .S3_protocol import S3Job, S3ProtocolError, _text, _Agent, _DEFAULT_AGENT
from . import training_constants as wire
from .fixed_constants import FixedAlias, constant_buffer, constant_bytes, constant_text

_INVALID_COMPACT_TRAINING_THRESHOLD = FixedAlias('training.validation.batch_header_compact_bound_invalid')
_BAD_DEVICE_SEED = 'S7 device seed must contain eight bytes'
_BAD_RESULT = FixedAlias('training.validation.batch_submission_fields_invalid')
_BAD_ACCOUNT = FixedAlias('training.validation.batch_account_identity_whitespace')
_BAD_RATE = 'training rate must be a nonnegative finite number in H/s'
_UPDATE_PAYLOAD = FixedAlias('training.label.batch_update_payload')
_UPDATE_SIZE_ERROR = FixedAlias('training.validation.batch_update_payload_length')
_UPDATE_SCHEMA_ERROR = FixedAlias('training.validation.batch_update_schema')
_BAD_SERVER_SEED = 'Unsupported S7 server seed schema'
_BAD_VERSION = FixedAlias('training.validation.batch_generation_attestation_revision')
_BATCH_TRAINING_THRESHOLD = FixedAlias('training.label.batch_acceptance_bound')
_MISSING_RESULT = FixedAlias('training.validation.batch_and_compact_attestation_required')
_BAD_COMPRESSION = FixedAlias('training.validation.batch_submission_compression_unsupported')
_SEED_MISMATCH = 'S7 share device seed does not match its job'

_HEADER_MASK = FixedAlias('training.format.batch_header_mask')

_WORK_PREFIX_MASK = (
    FixedAlias('training.format.batch_update_prefix_mask')
)

_HEX = re.compile('[0-9a-fA-F]+\\Z')


def header_mask():
    return constant_bytes(_HEADER_MASK)


def work_prefix_mask():
    return constant_bytes(_WORK_PREFIX_MASK)

class UnknownTrainingRequest(S3ProtocolError):
    pass

def _hex_bytes(value: Any, length: int, label: str) -> bytes:
    if not isinstance(value, str) or len(value) != length * 2 or _HEX.fullmatch(value) is None:
        prefix = constant_text(FixedAlias('training.label.legacy_batch_protocol_prefix'))
        raise S3ProtocolError(f'{prefix}{label} must contain exactly {length} hexadecimal bytes')
    return bytes.fromhex(value)

def decode_update_payload(value: str) -> bytes:
    payload = _hex_bytes(value, 200, constant_text(_UPDATE_PAYLOAD))
    with constant_buffer(_WORK_PREFIX_MASK) as prefix, constant_buffer(_HEADER_MASK) as mask:
        header = bytes((byte ^ prefix[index] ^ mask[index % 32] for index, byte in enumerate(payload[:76])))
    compact_limit = int.from_bytes(header[72:76], 'little')
    exponent, mantissa = (compact_limit >> 24, compact_limit & 8388607)
    training_threshold = mantissa >> 8 * (3 - exponent) if exponent <= 3 else mantissa << 8 * (exponent - 3)
    if compact_limit & 8388608 or not 0 < training_threshold < 1 << 256:
        raise S3ProtocolError(constant_text(_INVALID_COMPACT_TRAINING_THRESHOLD))
    return header

@dataclass(frozen=True)
class TrainingJob(S3Job):
    update_payload: bytes = b''
    device_seed: bytes | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cert_version != 3 or self.height != 0:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_generation_unsupported')))
        if not isinstance(self.update_payload, bytes) or len(self.update_payload) != 200:
            raise S3ProtocolError(constant_text(_UPDATE_SIZE_ERROR))
        if self.device_seed is not None and (not isinstance(self.device_seed, bytes) or len(self.device_seed) != 8):
            raise S3ProtocolError(_BAD_DEVICE_SEED)

@dataclass(frozen=True)
class TrainingSubmission:
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

class TrainingDialect:
    agent = _Agent()

    def __init__(self, training_identity: str | None, worker: str, agent: str=_DEFAULT_AGENT) -> None:
        self.training_identity = None if training_identity is None else _text(training_identity, 'WR_S2', 256)
        self.worker = _text(worker, wire.DEVICE_ID, 128, allow_empty=True)
        self._agent = _DEFAULT_AGENT if agent is _DEFAULT_AGENT else _text(agent, 'agent', 128)
        if any(character.isspace() for character in (training_identity or '') + worker):
            raise S3ProtocolError(constant_text(_BAD_ACCOUNT))

    def auth_requests(self, start_id: int=1) -> list[dict[str, Any]]:
        if self.training_identity is None:
            raise S3ProtocolError('WR login requires a configured identity')
        if type(start_id) is not int or not 0 <= start_id < 1 << 64:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_request_identifier_range')))
        return [{'id': start_id, 'method': wire.LOGIN, 'params': [self.training_identity, self.worker, self.agent]}]

    @staticmethod
    def report_request(gpu_name: str, work_rate: float, request_id: int) -> dict[str, Any]:
        if type(request_id) is not int or not 0 <= request_id < 1 << 64:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_request_identifier_range')))
        name = _text(gpu_name, 'gpu_name', 256)
        if not name.strip() or not name.isprintable():
            raise S3ProtocolError('gpu_name must contain printable GPU identification')
        if isinstance(work_rate, bool) or not isinstance(work_rate, (int, float)):
            raise S3ProtocolError(_BAD_RATE)
        try:
            rate = float(work_rate)
        except OverflowError:
            raise S3ProtocolError(_BAD_RATE) from None
        if not math.isfinite(rate) or rate < 0:
            raise S3ProtocolError(_BAD_RATE)
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
            raise UnknownTrainingRequest(constant_text(FixedAlias('training.validation.batch_server_method_unknown')))
        if not isinstance(response, Mapping) or set(response) - {'id', 'status', 'result', 'error'}:
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_server_fields_unknown')))
        request_id, status = (response.get('id'), response.get('status'))
        if type(request_id) is not int or not -1 <= request_id < 1 << 64 or type(status) is not int or (status not in (0, 1)):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_response_envelope_unknown')))
        result = response.get('result')
        if request_id >= 0 and type(result) is bool:
            return 'reply'
        if request_id != -1 or status != 1 or response.get('error') is not None or (not isinstance(result, list)) or (not result):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_server_message_shape_unknown')))
        if wire.text_matches('SERVER_SEED', result[0]):
            if len(result) != 2:
                raise S3ProtocolError(_BAD_SERVER_SEED)
            _hex_bytes(result[1], 8, wire.SERVER_SEED)
            return result[0]
        if result[0] == wire.JOB_UPDATE:
            if len(result) != 5 or not all((isinstance(field, str) for field in result)):
                raise S3ProtocolError(constant_text(_UPDATE_SCHEMA_ERROR))
            return wire.JOB_UPDATE
        if isinstance(result[0], str):
            raise UnknownTrainingRequest(constant_text(FixedAlias('training.validation.batch_server_notification_unknown')))
        raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_notification_name_malformed')))

    @staticmethod
    def parse_server_seed(result: list) -> bytes:
        if not isinstance(result, list) or len(result) != 2 or not wire.text_matches('SERVER_SEED', result[0]):
            raise S3ProtocolError(_BAD_SERVER_SEED)
        return _hex_bytes(result[1], 8, wire.SERVER_SEED)

    @staticmethod
    def parse_job(result: list, configured_certificate_version: int | None=None, device_seed: bytes | None=None) -> TrainingJob:
        if not isinstance(result, list) or len(result) != 5 or result[0] != wire.JOB_UPDATE or (not all((isinstance(field, str) for field in result))):
            raise S3ProtocolError(constant_text(_UPDATE_SCHEMA_ERROR))
        if result[4] != '3':
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_generation_unsupported')))
        if configured_certificate_version is not None and (type(configured_certificate_version) is not int or configured_certificate_version != 3):
            raise S3ProtocolError(constant_text(_BAD_VERSION))
        payload = _hex_bytes(result[1], 200, constant_text(_UPDATE_PAYLOAD))
        training_threshold_bytes = _hex_bytes(result[2], 32, constant_text(_BATCH_TRAINING_THRESHOLD))
        return TrainingJob(job_id=_text(result[3], wire.JOB_ID, 128), header_bytes=decode_update_payload(result[1]), height=0, cert_version=3, training_threshold=int.from_bytes(training_threshold_bytes, 'little'), encoded_training_threshold=result[2], update_payload=payload, device_seed=device_seed)

    @staticmethod
    def submission_request(job: TrainingJob, submission: TrainingSubmission, request_id: int, negotiated_gzip: bool=False) -> dict[str, Any]:
        if not isinstance(job, TrainingJob) or not isinstance(submission, TrainingSubmission):
            raise S3ProtocolError(constant_text(_MISSING_RESULT))
        if negotiated_gzip:
            raise S3ProtocolError(constant_text(_BAD_COMPRESSION))
        if job.device_seed is None or submission.device_seed != job.device_seed:
            raise S3ProtocolError(_SEED_MISMATCH)
        return submission.request(request_id)
