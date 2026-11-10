from workrunner.fixed_constants import FixedAlias, constant_text
from workrunner.training_records import encode_varint
from job_gateway.record_format.training_header import TrainingHeader
from job_gateway.record_format.training_certificate import TrainingCertificate

class TrainingEnvelope:

    def __init__(self, header: TrainingHeader, raw_records: list[bytes], training_certificate: TrainingCertificate):
        self.header = header
        if header.proof_commitment is None:
            header.proof_commitment = training_certificate.get_proof_commitment()
        elif header.proof_commitment != training_certificate.get_proof_commitment():
            raise ValueError(constant_text(FixedAlias('training.validation.attestation_commitment_mismatch')))
        self.raw_records = raw_records
        self.training_certificate = training_certificate

    def serialize(self) -> bytes:
        training_certificate_bytes = self.training_certificate.serialize()
        header_bytes = self.header.serialize()
        record_count_bytes = encode_varint(len(self.raw_records))
        records_bytes = b''.join(self.raw_records)
        return training_certificate_bytes + header_bytes + record_count_bytes + records_bytes
