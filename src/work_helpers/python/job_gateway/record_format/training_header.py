from workrunner.fixed_constants import FixedAlias, constant_text
from dataclasses import dataclass
from workrunner.protocol_types import PartialHeader
PROOF_COMMITMENT_SIZE = 32

@dataclass
class TrainingHeader:
    incomplete_header: PartialHeader
    proof_commitment: bytes | None = None

    def __post_init__(self):
        if self.proof_commitment is not None and len(self.proof_commitment) != PROOF_COMMITMENT_SIZE:
            prefix = constant_text(FixedAlias('training.validation.attestation_commitment_length_prefix'))
            raise ValueError(prefix + f'{PROOF_COMMITMENT_SIZE} bytes, got {len(self.proof_commitment)}')

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
    def training_threshold_bits(self) -> int:
        return self.incomplete_header.compact_limit

    @classmethod
    def get_serialized_header_size(cls) -> int:
        return PartialHeader.SERIALIZED_SIZE + PROOF_COMMITMENT_SIZE

    @classmethod
    def deserialize(cls, data: bytes) -> 'TrainingHeader':
        header_size = PartialHeader.SERIALIZED_SIZE
        header = PartialHeader.from_bytes(data[:header_size])
        proof_commitment = data[header_size:header_size + PROOF_COMMITMENT_SIZE]
        return cls(incomplete_header=header, proof_commitment=proof_commitment)

    def serialize_without_proof_commitment(self) -> bytes:
        return bytes(self.incomplete_header.to_bytes())

    def serialize(self) -> bytes:
        if self.proof_commitment is None:
            raise ValueError(constant_text(FixedAlias('training.validation.attestation_commitment_missing')))
        return self.serialize_without_proof_commitment() + self.proof_commitment
