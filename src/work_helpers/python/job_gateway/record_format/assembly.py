from workrunner.fixed_constants import constant_text
from hashlib import sha256
from workrunner.address import Encoding, decode_address, convertbits
from workrunner.records import Script
from workrunner.records import Record, RecordInput, RecordOutput, AttestationStack

def double_digest(data: bytes) -> bytes:
    return sha256(sha256(data).digest()).digest()

def calculate_tree_root(record_ids: list[str]) -> bytes:
    tx_hashes = [bytes.fromhex(record_id)[::-1] for record_id in record_ids]
    return _compute_tree_root(tx_hashes)

def _compute_tree_root(hashes: list[bytes]) -> bytes:
    assert len(hashes) > 0, constant_text(b'>BxuA#?;Tk)Q8i^hs4Or*V5C%hswg$%*(=u)yT-x')
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

def bits_to_target(bits: int) -> int:
    exponent = bits >> 24 & 255
    mantissa = bits & 16777215
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF or bits & 0x800000:
        raise ValueError(constant_text(b'?#a={%g)1x#K_9m#>CTy)5g`n!qW'))
    target = mantissa << (8 * (exponent - 3)) if exponent >= 3 else mantissa >> (8 * (3 - exponent))
    return target

_EMPTY_REFERENCE = int(constant_text(b'!NI}7!NI}7'), 16)

def build_reward_record(height: int, reward_value: int, S1_address: str, reward_aux: dict[str, str] | None=None, default_attestation_commitment: str | None=None) -> Record:
    output_script = address_to_output_script(S1_address)
    height_script = Script([height])
    reward_script_bytes = bytes.fromhex(height_script.to_hex())
    reward_script_bytes += b'\x00'
    if reward_aux and 'flags' in reward_aux:
        aux_flags = bytes.fromhex(reward_aux['flags'])
        reward_script_bytes += aux_flags
    reward_input = RecordInput(record_id=bytes(32).hex(), txout_index=_EMPTY_REFERENCE, script_sig=Script.from_raw(reward_script_bytes))
    reward_output = RecordOutput(reward_value, output_script)
    outputs = [reward_output]
    has_attestation = default_attestation_commitment is not None
    if has_attestation:
        attestation_script = Script([constant_text(b'==b>b;`8$L=>'), constant_text(b'#>SPF#-75%') + default_attestation_commitment])
        attestation_output = RecordOutput(0, attestation_script)
        outputs.append(attestation_output)
    reward_tx = Record(inputs=[reward_input], outputs=outputs, locktime=b'\x00\x00\x00\x00', version=b'\x01\x00\x00\x00', tagged=has_attestation, attestations=[AttestationStack([bytes(32).hex()])] if has_attestation else None)
    return reward_tx

def address_to_output_script(address: str) -> Script:
    hrp, data, encoding = decode_address(address)
    if hrp is None or data is None:
        raise ValueError(constant_text(b'?#a={%g)1x#lpnTl$D6Z!o<&%mCA?4!^73W)YJ'))
    if len(data) == 0:
        raise ValueError(constant_text(b'?#a={%g)1x(9YAz!qn7<*VV|t)yB$)kHX5=)82=}#?!`*'))
    attestation_version = data[0]
    if attestation_version != 1:
        raise ValueError(constant_text(b';@{W8#M8pVhx5kQ)yT-xhtST`$->mshta~-)XvDshn9<nz{t~w') + str(attestation_version))
    if encoding != Encoding.EXTENDED:
        raise ValueError(constant_text(b'^TyZJ$jH-&#>2za!qn7<%F@)+hsDB&#lpnTl$FYhhrr0whX') + constant_text(b';>pCw!_LXTiRI$t@067'))
    attestation_program_5bit = data[1:]
    attestation_program_8bit = convertbits(attestation_program_5bit, 5, 8, False)
    if attestation_program_8bit is None:
        raise ValueError(constant_text(b'?#a={%g)1x(9YAz!qn7<*VV|t)yB$)kHpBy(Zbc!#m>{zhr!0q%fiEs'))
    attestation_program_bytes = bytes(attestation_program_8bit)
    if len(attestation_program_bytes) != 32:
        raise ValueError(constant_text(b'^TyZJ$jH-&(9YAz!qn7<*VV|t)yB$)%F@)+hsDB&l$D3Y-qXU=i-*9-(}w') + str(len(attestation_program_bytes)))
    script_bytes = b'Q ' + attestation_program_bytes
    return Script.from_raw(script_bytes.hex())
