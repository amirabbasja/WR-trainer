from copy import copy
from miner_utils import get_logger
from LRAEP_trainer.primitives import PlainProof, generate_proof_for_cert_version, verify_proof_for_cert_version
from LRAEP_gateway.blockchain_utils.LRAEP_block import LRAEPBlock
from LRAEP_gateway.blockchain_utils.zk_certificate import ZKCertificate
from LRAEP_gateway.comm.dataclasses import BlockTemplate
_LOGGER = get_logger(__name__)

class ProofGenerator:

    @classmethod
    def generate_block(cls, plain_proof: PlainProof, template: BlockTemplate, debug_mode: bool=False) -> LRAEPBlock:
        _LOGGER.debug('Generating block from PlainProof')
        cert_version = template.required_cert_version
        zk_proof = generate_proof_for_cert_version(cert_version, template.header.incomplete_header, plain_proof)
        _LOGGER.debug('Generated ZK proof')
        if debug_mode:
            _LOGGER.info('verifying ZK proof')
            result, msg = verify_proof_for_cert_version(cert_version, template.header.incomplete_header, zk_proof)
            if not result:
                raise AssertionError(f'Failed to verify proof: {msg}')
            _LOGGER.info('verified ZK proof')
        header = copy(template.header)
        zk_certificate = ZKCertificate.from_LRAEP_header(header, zk_proof, cert_version=cert_version)
        block = LRAEPBlock(header=header, raw_txns=template.get_raw_transactions(), zk_certificate=zk_certificate)
        _LOGGER.debug('Generated block')
        return block
