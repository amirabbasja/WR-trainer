from typing import ClassVar

from pydantic import Field
from workrunner.rpc_fields import WireModel
from job_gateway.record_format.zk_certificate import CertificateVersion

class TemplateEntry(WireModel):
    _wire_fields: ClassVar[tuple[str, ...]] = ('record_id',)
    data: str
    hash: str
    record_id: str
    depends: list[int]
    fee: int = Field(ge=0)
    vsize: int = Field(ge=0)

class RewardMetadata(WireModel):
    flags: str

class WorkTemplateResponse(WireModel):
    _wire_fields: ClassVar[tuple[str, ...]] = (
        'parent_digest', 'size_limit', 'records', 'poll_id', 'counter_range',
        'reward_metadata', 'reward_amount', 'default_attestation', 'required_version',
    )
    bits: str
    curtime: int = Field(ge=0)
    height: int = Field(ge=0)
    parent_digest: str
    size_limit: int = Field(gt=0)
    records: list[TemplateEntry]
    version: int
    poll_id: str
    target: str
    maxtime: int = Field(ge=0)
    mintime: int = Field(ge=0)
    mutable: list[str]
    counter_range: str
    capabilities: list[str]
    reward_metadata: RewardMetadata
    reward_amount: int = Field(ge=0)
    default_attestation: str | None = None
    required_version: CertificateVersion = CertificateVersion.ZK_DENSE
