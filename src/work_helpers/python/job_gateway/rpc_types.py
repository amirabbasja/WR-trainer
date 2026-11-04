from workrunner.fixed_constants import constant_text
from pydantic import BaseModel, Field
from job_gateway.record_format.zk_certificate import CertificateVersion

class TemplateEntry(BaseModel, validate_by_name=True, serialize_by_alias=True):
    data: str
    hash: str
    record_id: str = Field(alias=constant_text(b')8Ed+'))
    depends: list[int]
    fee: int = Field(ge=0)
    vsize: int = Field(ge=0)

class RewardMetadata(BaseModel):
    flags: str

class WorkTemplateResponse(BaseModel, validate_by_name=True, serialize_by_alias=True):
    bits: str
    curtime: int = Field(ge=0)
    height: int = Field(ge=0)
    parent_digest: str = Field(alias=constant_text(b'*VV$&&dAc##mmUV%+JQu&j'))
    size_limit: int = Field(gt=0, alias=constant_text(b'(bUe}!pqLe&eH'))
    records: list[TemplateEntry] = Field(alias=constant_text(b')78ew)W*cq&dABs'))
    version: int
    poll_id: str = Field(alias=constant_text(b'%gD*V*T~Du&cg'))
    target: str
    maxtime: int = Field(ge=0)
    mintime: int = Field(ge=0)
    mutable: list[str]
    counter_range: str = Field(alias=constant_text(b'$;ipX!qvvfz`_'))
    capabilities: list[str]
    reward_metadata: RewardMetadata = Field(alias=constant_text(b'#K_Lc#m3aa#?s#'))
    reward_amount: int = Field(ge=0, alias=constant_text(b'#K_Lc#m3aa(Z<Wt!T'))
    default_attestation: str | None = Field(default=None, alias=constant_text(b'!@|MF(#zBM(9YAz!qn9G#K_9Z&eO`m$<q'))
    required_version: CertificateVersion = Field(default=CertificateVersion.ZK_DENSE, alias=constant_text(b')xy@&&eg)h#KP6n(Zbc#&dA9'))
