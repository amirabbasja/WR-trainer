from workrunner.fixed_constants import FixedAlias, constant_text
from hashlib import sha256
from workrunner.training_identity import Encoding, decode_training_identity, convertbits
from workrunner.training_records import TrainingPayload
from workrunner.training_records import TrainingRecord, TrainingInput, TrainingOutput, TrainingAttestation

def double_digest(data: bytes) -> bytes:
    return sha256(sha256(data).digest()).digest()

def calculate_tree_root(record_ids: list[str]) -> bytes:
    record_digests = [bytes.fromhex(record_id)[::-1] for record_id in record_ids]
    return _compute_tree_root(record_digests)

def _compute_tree_root(hashes: list[bytes]) -> bytes:
    assert len(hashes) > 0, 'No hashes to compute matrix tree root'
    if len(hashes) == 1:
        return hashes[0][::-1]
    current_level = hashes.copy()
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                combined = current_level[i] + current_level[i + 1]
            else:
                combined = current_level[i] + current_level[i]
            next_level.append(double_digest(combined))
        current_level = next_level
    return current_level[0][::-1]

def bits_to_training_threshold(bits: int) -> int:
    exponent = bits >> 24 & 255
    mantissa = bits & 16777215
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF or bits & 0x800000:
        raise ValueError(constant_text(FixedAlias('training.validation.compact_acceptance_bound_invalid')))
    training_threshold = mantissa << (8 * (exponent - 3)) if exponent >= 3 else mantissa >> (8 * (3 - exponent))
    return training_threshold

_EMPTY_REFERENCE = int(constant_text(FixedAlias('training.format.empty_parent_output_index')), 16)

def build_initial_record(height: int, training_credit: int, training_identity: str, training_metadata: dict[str, str] | None=None, default_attestation_commitment: str | None=None) -> TrainingRecord:
    output_payload = identity_to_output_payload(training_identity)
    step_payload = TrainingPayload([height])
    initial_payload_bytes = bytes.fromhex(step_payload.to_hex())
    initial_payload_bytes += b'\x00'
    if training_metadata and 'flags' in training_metadata:
        aux_flags = bytes.fromhex(training_metadata['flags'])
        initial_payload_bytes += aux_flags
    initial_input = TrainingInput(record_id=bytes(32).hex(), source_output_index=_EMPTY_REFERENCE, input_payload=TrainingPayload.from_raw(initial_payload_bytes))
    initial_output = TrainingOutput(training_credit, output_payload)
    outputs = [initial_output]
    has_attestation = default_attestation_commitment is not None
    if has_attestation:
        payload_tag = constant_text(FixedAlias('training.format.attestation_payload_tag'))
        attestation_payload = TrainingPayload(['training.attestation', payload_tag + default_attestation_commitment])
        attestation_output = TrainingOutput(0, attestation_payload)
        outputs.append(attestation_output)
    initial_record = TrainingRecord(inputs=[initial_input], outputs=outputs, release_step=b'\x00\x00\x00\x00', version=b'\x01\x00\x00\x00', tagged=has_attestation, attestations=[TrainingAttestation([bytes(32).hex()])] if has_attestation else None)
    return initial_record

def identity_to_output_payload(identity: str) -> TrainingPayload:
    """Convert a version-1 training account identity into the gateway's record output."""
    hrp, data, encoding = decode_training_identity(identity)
    if hrp is None or data is None:
        raise ValueError('Invalid training identity checksum')
    if len(data) == 0:
        raise ValueError('Invalid identity payload (empty data)')
    attestation_version = data[0]
    if attestation_version != 1:
        raise ValueError('Expected identity payload version 1, got ' + str(attestation_version))
    if encoding != Encoding.EXTENDED:
        raise ValueError(f'Version-1 training identity requires the extended checksum, got {encoding.name}')
    attestation_program_5bit = data[1:]
    attestation_program_8bit = convertbits(attestation_program_5bit, 5, 8, False)
    if attestation_program_8bit is None:
        raise ValueError('Invalid identity payload (convertbits failed)')
    attestation_program_bytes = bytes(attestation_program_8bit)
    if len(attestation_program_bytes) != 32:
        raise ValueError('Version-1 identity payload must be 32 bytes, got ' + str(len(attestation_program_bytes)))
    # The output starts with the version-1 opcode and the 32-byte push length.
    return TrainingPayload.from_raw(b'\x51\x20' + attestation_program_bytes)
