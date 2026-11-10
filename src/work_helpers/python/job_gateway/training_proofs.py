from workrunner.fixed_constants import FixedAlias, constant_text
from copy import copy
from S5_utils import get_logger
from workrunner.protocol_types import PlainProof, generate_proof_for_cert_version, verify_proof_for_cert_version
from job_gateway.record_format.training_record import TrainingEnvelope
from job_gateway.record_format.training_certificate import TrainingCertificate
from job_gateway.comm.training_types import TrainingTemplate
_LOGGER = get_logger(__name__)

class TrainingProofGenerator:

    @classmethod
    def generate_record(cls, plain_proof: PlainProof, template: TrainingTemplate, debug_mode: bool=False) -> TrainingEnvelope:
        _LOGGER.debug(constant_text(FixedAlias('training.log.attestation_record_assembly')))
        cert_version = template.required_cert_version
        training_proof = generate_proof_for_cert_version(cert_version, template.header.incomplete_header, plain_proof)
        _LOGGER.debug(constant_text(FixedAlias('training.log.zero_knowledge_attestation_generated')))
        if debug_mode:
            _LOGGER.info(constant_text(FixedAlias('training.log.zero_knowledge_attestation_verifying')))
            result, msg = verify_proof_for_cert_version(cert_version, template.header.incomplete_header, training_proof)
            if not result:
                raise AssertionError(f'Failed to verify proof: {msg}')
            _LOGGER.info(constant_text(FixedAlias('training.log.zero_knowledge_attestation_verified')))
        header = copy(template.header)
        training_certificate = TrainingCertificate.from_training_header(header, training_proof, cert_version=cert_version)
        training_record = TrainingEnvelope(header=header, raw_records=template.get_raw_records(), training_certificate=training_certificate)
        _LOGGER.debug(constant_text(FixedAlias('training.log.attestation_record_generated')))
        return training_record
