from pydantic import BaseModel, Field
from LRAEP_gateway.blockchain_utils.zk_certificate import CertificateVersion

class BlockTemplateTx(BaseModel):
    data: str
    hash: str
    txid: str
    depends: list[int]
    fee: int = Field(ge=0)
    vsize: int = Field(ge=0)

class CoinbaseAux(BaseModel):
    flags: str

class GetBlockTemplateResponse(BaseModel):
    bits: str
    curtime: int = Field(ge=0)
    height: int = Field(ge=0)
    previousblockhash: str
    vsizelimit: int = Field(gt=0)
    transactions: list[BlockTemplateTx]
    version: int
    longpollid: str
    target: str
    maxtime: int = Field(ge=0)
    mintime: int = Field(ge=0)
    mutable: list[str]
    noncerange: str
    capabilities: list[str]
    coinbaseaux: CoinbaseAux
    coinbasevalue: int = Field(ge=0)
    default_witness_commitment: str | None = None
    requiredcertversion: CertificateVersion = CertificateVersion.ZK_DENSE
