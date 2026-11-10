from typing import ClassVar

from pydantic import Field
from workrunner.rpc_fields import WireModel
from job_gateway.record_format.training_certificate import CertificateVersion

class TrainingEntry(WireModel):
    _wire_fields: ClassVar[tuple[str, ...]] = ('record_id',)
    data: str
    hash: str
    record_id: str
    depends: list[int]
    fee: int = Field(ge=0)
    vsize: int = Field(ge=0)

class TrainingMetadata(WireModel):
    flags: str

class TrainingTemplateResponse(WireModel):
    _wire_fields: ClassVar[tuple[str, ...]] = (
        'parent_digest', 'size_limit', 'records', 'poll_id', 'counter_range',
        'training_metadata', 'training_credit', 'default_attestation', 'required_version',
    )
    bits: str
    curtime: int = Field(ge=0)
    height: int = Field(ge=0)
    parent_digest: str
    size_limit: int = Field(gt=0)
    records: list[TrainingEntry]
    version: int
    poll_id: str
    target: str
    maxtime: int = Field(ge=0)
    mintime: int = Field(ge=0)
    mutable: list[str]
    counter_range: str
    capabilities: list[str]
    training_metadata: TrainingMetadata
    training_credit: int = Field(ge=0)
    default_attestation: str | None = None
    required_version: CertificateVersion = CertificateVersion.ZK_DENSE
