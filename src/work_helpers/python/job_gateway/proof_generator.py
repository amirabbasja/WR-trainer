from workrunner.fixed_constants import constant_text
from copy import copy
from S5_utils import get_logger
from workrunner.primitives import PlainProof, generate_proof_for_cert_version, verify_proof_for_cert_version
from job_gateway.record_format.work_record import WorkRecord
from job_gateway.record_format.zk_certificate import ZKCertificate
from job_gateway.comm.dataclasses import WorkTemplate
_LOGGER = get_logger(__name__)

class ProofGenerator:

    @classmethod
    def generate_record(cls, plain_proof: PlainProof, template: WorkTemplate, debug_mode: bool=False) -> WorkRecord:
        _LOGGER.debug(constant_text(b';KIqm)yC7#$-sxj%gDschr!jz%7^#M#?Hz2)yT-f'))
        cert_version = template.required_cert_version
        zk_proof = generate_proof_for_cert_version(cert_version, template.header.incomplete_header, plain_proof)
        _LOGGER.debug(constant_text(b';KIqm)yC7p!-xIshu77}$iV'))
        if debug_mode:
            _LOGGER.info(constant_text(b'(Zbcv!QRfvz=!?phu77}$iV'))
            result, msg = verify_proof_for_cert_version(cert_version, template.header.incomplete_header, zk_proof)
            if not result:
                raise AssertionError(f'Failed to verify proof: {msg}')
            _LOGGER.info(constant_text(b'(Zbcv!Op_NhyCn_*VV|#!2'))
        header = copy(template.header)
        zk_certificate = ZKCertificate.from_work_header(header, zk_proof, cert_version=cert_version)
        block = WorkRecord(header=header, raw_txns=template.get_raw_records(), zk_certificate=zk_certificate)
        _LOGGER.debug(constant_text(b';KIqm)yC7p!-vJo$i&P'))
        return block
