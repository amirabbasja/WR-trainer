from workrunner.fixed_constants import constant_text
from workrunner.records import encode_varint
from job_gateway.record_format.work_header import WorkHeader
from job_gateway.record_format.zk_certificate import ZKCertificate

class WorkRecord:

    def __init__(self, header: WorkHeader, raw_txns: list[bytes], zk_certificate: ZKCertificate):
        self.header = header
        if header.proof_commitment is None:
            header.proof_commitment = zk_certificate.get_proof_commitment()
        elif header.proof_commitment != zk_certificate.get_proof_commitment():
            raise ValueError(constant_text(b'_tnVA!H2}i%F52u%EHOhhsw^>%Er^g&j'))
        self.raw_txns = raw_txns
        self.zk_certificate = zk_certificate

    def serialize(self) -> bytes:
        zk_certificate_bytes = self.zk_certificate.serialize()
        header_bytes = self.header.serialize()
        tx_count_bytes = encode_varint(len(self.raw_txns))
        records_bytes = b''.join(self.raw_txns)
        return zk_certificate_bytes + header_bytes + tx_count_bytes + records_bytes
