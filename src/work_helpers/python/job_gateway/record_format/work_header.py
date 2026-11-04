from workrunner.fixed_constants import constant_text
from dataclasses import dataclass
from workrunner.primitives import PartialHeader
PROOF_COMMITMENT_SIZE = 32

@dataclass
class WorkHeader:
    incomplete_header: PartialHeader
    proof_commitment: bytes | None = None

    def __post_init__(self):
        if self.proof_commitment is not None and len(self.proof_commitment) != PROOF_COMMITMENT_SIZE:
            raise ValueError(constant_text(b'_tnVA!H2}i%F52u%EHOhhsx5_(}%^vhX') + f'{PROOF_COMMITMENT_SIZE} bytes, got {len(self.proof_commitment)}')

    @property
    def version(self) -> int:
        return self.incomplete_header.version

    @property
    def parent_digest(self) -> bytes:
        return bytes(self.incomplete_header.parent_digest)

    @property
    def tree_root(self) -> bytes:
        return bytes(self.incomplete_header.tree_root)

    @property
    def timestamp(self) -> int:
        return self.incomplete_header.timestamp

    @property
    def target_bits(self) -> int:
        return self.incomplete_header.compact_limit

    @classmethod
    def get_serialized_header_size(cls) -> int:
        return PartialHeader.SERIALIZED_SIZE + PROOF_COMMITMENT_SIZE

    @classmethod
    def deserialize(cls, data: bytes) -> 'WorkHeader':
        header_size = PartialHeader.SERIALIZED_SIZE
        header = PartialHeader.from_bytes(data[:header_size])
        proof_commitment = data[header_size:header_size + PROOF_COMMITMENT_SIZE]
        return cls(incomplete_header=header, proof_commitment=proof_commitment)

    def serialize_without_proof_commitment(self) -> bytes:
        return bytes(self.incomplete_header.to_bytes())

    def serialize(self) -> bytes:
        if self.proof_commitment is None:
            raise ValueError(constant_text(b'_tnVA!H2}i%F52u%EHOhhtAZ8$;i`()WXv'))
        return self.serialize_without_proof_commitment() + self.proof_commitment
