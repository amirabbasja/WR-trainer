import base64
from dataclasses import dataclass
from typing import Any, ClassVar
import torch
from workrunner.training_records import TrainingRecord
from job_gateway.record_format.training_assembly import bits_to_training_threshold, calculate_tree_root, build_initial_record
from job_gateway.record_format.training_header import TrainingHeader
from job_gateway.record_format.training_certificate import CertificateVersion
from job_gateway.comm.S1_configuration import S1Configuration, MoEConfig, WorkConfigurationFactory
from job_gateway.training_responses import TrainingTemplateResponse
from workrunner import training_constants as wire
from workrunner.protocol_types import TRAINING_BASE_RANK, PartialHeader, scaled_training_threshold

def get_bytes(data: str | bytes) -> bytes:
    if isinstance(data, str):
        return bytes.fromhex(data)
    return data

def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')

def b64_decode(data: str) -> bytes:
    return base64.b64decode(data.encode('ascii'))

def decode_dtype(encoded_dtype: str) -> torch.dtype:
    return getattr(torch, encoded_dtype.replace('torch.', ''))

@dataclass
class TrainingTemplate:
    header: TrainingHeader
    height: int
    raw_records: list[bytes]
    initial_record: TrainingRecord
    required_cert_version: CertificateVersion

    @classmethod
    def from_template_response(cls, data: TrainingTemplateResponse, training_identity: str) -> 'TrainingTemplate':
        parent_digest = data.parent_digest
        version = data.version
        bits = data.bits
        curtime = data.curtime
        initial_record = build_initial_record(height=data.height, training_credit=data.training_credit, training_identity=training_identity, training_metadata=data.training_metadata.model_dump(), default_attestation_commitment=data.default_attestation)
        raw_records = [bytes.fromhex(entry.data) for entry in data.records]
        record_ids = [entry.record_id for entry in data.records]
        initial_record_id = initial_record.get_record_id()
        tree_root = calculate_tree_root([initial_record_id] + record_ids)
        height = data.height
        bits_translation = bits_to_training_threshold(int(bits, 16))
        if int(data.target, 16) != bits_translation:
            raise ValueError(f'target and bits must match: {data.target} != {bits_translation}')
        return cls(header=TrainingHeader(incomplete_header=PartialHeader(version=version, parent_digest=bytes.fromhex(parent_digest), tree_root=tree_root, timestamp=curtime, compact_limit=int(bits, 16))), height=height, raw_records=raw_records, initial_record=initial_record, required_cert_version=CertificateVersion(data.required_version))

    def get_raw_records(self) -> list[bytes]:
        initial_bytes = self.initial_record.to_bytes(self.initial_record.tagged)
        return [initial_bytes] + self.raw_records

    @property
    def bits(self) -> int:
        return self.header.training_threshold_bits

    @property
    def training_threshold(self) -> int:
        return bits_to_training_threshold(self.bits)

@dataclass
class CommitmentHash:
    noise_seed_A: bytes
    noise_seed_B: bytes

    def to_dict(self) -> dict[str, Any]:
        return {'noise_seed_A': b64_encode(self.noise_seed_A), 'noise_seed_B': b64_encode(self.noise_seed_B)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CommitmentHash':
        return cls(noise_seed_A=b64_decode(data['noise_seed_A']), noise_seed_B=b64_decode(data['noise_seed_B']))

@dataclass
class MoEBlockInfo:
    expert_index: int
    num_experts: int
    n_per_expert: int
    top_k: int
    inner_a_rows: list[int]
    inner_b_cols: list[int]
    routing_data: torch.Tensor
    expert_routing_offsets: list[int]

@dataclass
class OpenedBlockInfo:
    A_row_indices: list[int]
    B_column_indices: list[int]
    A: torch.Tensor | None
    B_t: torch.Tensor | None
    commitment_hash: CommitmentHash | None
    noise_rank: int
    moe: MoEBlockInfo | None = None
    noise_range: ClassVar[int] = 128

    def get_S1_config(self) -> S1Configuration:
        if self.A is None or self.B_t is None:
            raise ValueError('A and B must be provided')
        if self.moe is not None:
            return WorkConfigurationFactory.create(common_dim=self.A.shape[1], rank=self.noise_rank, row_indices=self.moe.inner_a_rows, col_indices=self.moe.inner_b_cols, moe=MoEConfig(e=self.moe.num_experts, top_k=self.moe.top_k))
        return WorkConfigurationFactory.create(common_dim=self.A.shape[1], rank=self.noise_rank, row_indices=self.A_row_indices, col_indices=self.B_column_indices)

@dataclass
class S1Job:
    incomplete_header_bytes: bytes
    training_threshold: int
    cert_version: CertificateVersion

    def to_dict(self) -> dict[str, Any]:
        return {'incomplete_header_bytes': b64_encode(self.incomplete_header_bytes), wire.TRAINING_THRESHOLD: self.training_threshold, wire.FORMAT_FIELD: int(self.cert_version)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'S1Job':
        return cls(incomplete_header_bytes=b64_decode(data['incomplete_header_bytes']), training_threshold=data[wire.TRAINING_THRESHOLD], cert_version=CertificateVersion(data[wire.FORMAT_FIELD]))

    @classmethod
    def from_template(cls, template: TrainingTemplate) -> 'S1Job':
        return cls(incomplete_header_bytes=template.header.serialize_without_proof_commitment(), training_threshold=template.training_threshold, cert_version=template.required_cert_version)

    def adjust_training_threshold(self, S1_config: S1Configuration) -> int:
        if S1_config.rank < TRAINING_BASE_RANK:
            raise ValueError(f'noise rank {S1_config.rank} is below the minimum {TRAINING_BASE_RANK}; batches would be rejected by the protocol')
        bound = scaled_training_threshold(self.training_threshold, S1_config)
        if bound is None:
            raise ValueError(f'no scaled training threshold for self.training_threshold={self.training_threshold!r} at rank {S1_config.rank}: degenerate config or training threshold too large')
        return bound

class S1PausedError(Exception):
    code = -32001
    message = 'S1_paused'

    def __init__(self, details: str=''):
        self.details = details
        super().__init__(f'{self.message}: {details}' if details else self.message)
