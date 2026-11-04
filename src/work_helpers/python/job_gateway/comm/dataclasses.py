import base64
from dataclasses import dataclass
from typing import Any, ClassVar
import torch
from workrunner.records import Record
from job_gateway.record_format.assembly import bits_to_target, calculate_tree_root, build_reward_record
from job_gateway.record_format.work_header import WorkHeader
from job_gateway.record_format.zk_certificate import CertificateVersion
from job_gateway.comm.S1_configuration import S1Configuration, MoEConfig, WorkConfigurationFactory
from job_gateway.rpc_types import WorkTemplateResponse
from workrunner import wire_constants as wire
from workrunner.primitives import PENALTY_BASE_RANK, PartialHeader, penalized_target_bound

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
class WorkTemplate:
    header: WorkHeader
    height: int
    raw_records: list[bytes]
    reward_tx: Record
    required_cert_version: CertificateVersion

    @classmethod
    def from_template_response(cls, data: WorkTemplateResponse, S1_address: str) -> 'WorkTemplate':
        parent_digest = data.parent_digest
        version = data.version
        bits = data.bits
        curtime = data.curtime
        reward_tx = build_reward_record(height=data.height, reward_value=data.reward_amount, S1_address=S1_address, reward_aux=data.reward_metadata.model_dump(), default_attestation_commitment=data.default_attestation)
        raw_records = [bytes.fromhex(tx.data) for tx in data.records]
        record_ids = [tx.record_id for tx in data.records]
        reward_record_id = reward_tx.get_record_id()
        tree_root = calculate_tree_root([reward_record_id] + record_ids)
        height = data.height
        bits_translation = bits_to_target(int(bits, 16))
        if int(data.target, 16) != bits_translation:
            raise ValueError(f'target and bits must match: {data.target} != {bits_translation}')
        return cls(header=WorkHeader(incomplete_header=PartialHeader(version=version, parent_digest=bytes.fromhex(parent_digest), tree_root=tree_root, timestamp=curtime, compact_limit=int(bits, 16))), height=height, raw_records=raw_records, reward_tx=reward_tx, required_cert_version=CertificateVersion(data.required_version))

    def get_raw_records(self) -> list[bytes]:
        reward_bytes = self.reward_tx.to_bytes(self.reward_tx.tagged)
        return [reward_bytes] + self.raw_records

    @property
    def bits(self) -> int:
        return self.header.target_bits

    @property
    def target(self) -> int:
        return bits_to_target(self.bits)

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
    target: int
    cert_version: CertificateVersion

    def to_dict(self) -> dict[str, Any]:
        return {'incomplete_header_bytes': b64_encode(self.incomplete_header_bytes), wire.THRESHOLD: self.target, wire.FORMAT_FIELD: int(self.cert_version)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'S1Job':
        return cls(incomplete_header_bytes=b64_decode(data['incomplete_header_bytes']), target=data[wire.THRESHOLD], cert_version=CertificateVersion(data[wire.FORMAT_FIELD]))

    @classmethod
    def from_template(cls, template: WorkTemplate) -> 'S1Job':
        return cls(incomplete_header_bytes=template.header.serialize_without_proof_commitment(), target=template.target, cert_version=template.required_cert_version)

    def adjust_target(self, S1_config: S1Configuration) -> int:
        if S1_config.rank < PENALTY_BASE_RANK:
            raise ValueError(f'noise rank {S1_config.rank} is below the minimum {PENALTY_BASE_RANK}; blocks would be rejected by consensus')
        bound = penalized_target_bound(self.target, S1_config)
        if bound is None:
            raise ValueError(f'no penalized target for self.target={self.target!r} at rank {S1_config.rank}: degenerate config or target too easy')
        return bound

class S1PausedError(Exception):
    code = -32001
    message = 'S1_paused'

    def __init__(self, details: str=''):
        self.details = details
        super().__init__(f'{self.message}: {details}' if details else self.message)
