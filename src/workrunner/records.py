from .string_codec import customDecoder
from dataclasses import dataclass, field
from hashlib import sha256


_OPCODES = {
    customDecoder(bytes.fromhex('044f505f30')): 0x0,
    customDecoder(bytes.fromhex('084f505f46414c5345')): 0x0,
    customDecoder(bytes.fromhex('0a4f505f314e4547415445')): 0x4f,
    customDecoder(bytes.fromhex('094f505f52455455524e')): 0x6a,
    customDecoder(bytes.fromhex('074f505f54525545')): 0x51,
}
_OPCODE_PREFIX = customDecoder(bytes.fromhex('034f505f'))
_OPCODES.update({f'{_OPCODE_PREFIX}{i}': 0x50 + i for i in range(1, 17)})
_OPTIONAL_SECTION = bytes.fromhex(customDecoder(bytes.fromhex('0430303031')))

def encode_varint(value):
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError('Length must be an unsigned 64-bit integer')
    if value < 253:
        return bytes([value])
    if value <= 0xFFFF:
        return b'\xfd' + value.to_bytes(2, 'little')
    if value <= 0xFFFFFFFF:
        return b'\xfe' + value.to_bytes(4, 'little')
    return b'\xff' + value.to_bytes(8, 'little')


def _push(value):
    length = len(value)
    if length < 76:
        return bytes([length]) + value
    if length <= 255:
        return b'\x4c' + bytes([length]) + value
    if length <= 65535:
        return b'\x4d' + length.to_bytes(2, 'little') + value
    if length <= 0xFFFFFFFF:
        return b'\x4e' + length.to_bytes(4, 'little') + value
    raise ValueError('Script element is too large')


def _script_number(value):
    if value == 0:
        return b'\x00'
    if value == -1:
        return b'\x4f'
    if 1 <= value <= 16:
        return bytes([0x50 + value])
    negative = value < 0
    magnitude = abs(value)
    raw = magnitude.to_bytes((magnitude.bit_length() + 7) // 8, 'little')
    if raw[-1] & 0x80:
        raw += bytes([0x80 if negative else 0])
    elif negative:
        raw = raw[:-1] + bytes([raw[-1] | 0x80])
    return _push(raw)


class Script:
    def __init__(self, script=None):
        self.script = list(script or [])
        self._raw = None

    @classmethod
    def from_raw(cls, raw):
        result = cls()
        result._raw = bytes.fromhex(raw) if isinstance(raw, str) else bytes(raw)
        return result

    def to_bytes(self):
        if self._raw is not None:
            return self._raw
        result = bytearray()
        for value in self.script:
            if type(value) is int:
                result.extend(_script_number(value))
            elif isinstance(value, str) and value in _OPCODES:
                result.append(_OPCODES[value])
            elif isinstance(value, str):
                result.extend(_push(bytes.fromhex(value)))
            elif isinstance(value, bytes):
                result.extend(_push(value))
            else:
                raise ValueError('Invalid script element')
        return bytes(result)

    def to_hex(self):
        return self.to_bytes().hex()


@dataclass
class RecordInput:
    record_id: str
    txout_index: int
    script_sig: Script = field(default_factory=Script)
    sequence: bytes = b'\xff\xff\xff\xff'

    def to_bytes(self):
        record_id = bytes.fromhex(self.record_id)
        if len(record_id) != 32 or type(self.txout_index) is not int or not 0 <= self.txout_index <= 0xFFFFFFFF or len(self.sequence) != 4:
            raise ValueError(customDecoder(bytes.fromhex('19496e76616c6964207472616e73616374696f6e20696e707574')))
        script = self.script_sig.to_bytes()
        return record_id[::-1] + self.txout_index.to_bytes(4, 'little') + encode_varint(len(script)) + script + self.sequence


@dataclass
class RecordOutput:
    amount: int
    output_script: Script

    def to_bytes(self):
        if type(self.amount) is not int or not 0 <= self.amount <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError(customDecoder(bytes.fromhex('21496e76616c6964207472616e73616374696f6e206f757470757420616d6f756e74')))
        script = self.output_script.to_bytes()
        return self.amount.to_bytes(8, 'little') + encode_varint(len(script)) + script


@dataclass
class AttestationStack:
    stack: list[str]

    def to_bytes(self):
        items = [bytes.fromhex(item) for item in self.stack]
        return encode_varint(len(items)) + b''.join(encode_varint(len(item)) + item for item in items)


@dataclass
class Record:
    inputs: list[RecordInput]
    outputs: list[RecordOutput]
    locktime: bytes = bytes(4)
    version: bytes = b'\x01\x00\x00\x00'
    tagged: bool = False
    attestations: list[AttestationStack] | None = None

    def to_bytes(self, tagged=None):
        if len(self.version) != 4 or len(self.locktime) != 4 or not self.inputs or not self.outputs:
            raise ValueError(customDecoder(bytes.fromhex('32496e76616c6964207472616e73616374696f6e20686561646572206f7220656d70747920696e707574732f6f757470757473')))
        attestation = self.tagged if tagged is None else bool(tagged and self.tagged)
        result = self.version + (_OPTIONAL_SECTION if attestation else b'')
        result += encode_varint(len(self.inputs)) + b''.join(item.to_bytes() for item in self.inputs)
        result += encode_varint(len(self.outputs)) + b''.join(item.to_bytes() for item in self.outputs)
        if attestation:
            if self.attestations is None or len(self.attestations) != len(self.inputs):
                raise ValueError(customDecoder(bytes.fromhex('275769746e6573736573206d757374206d61746368207472616e73616374696f6e20696e70757473')))
            result += b''.join(item.to_bytes() for item in self.attestations)
        return result + self.locktime

    def get_record_id(self):
        payload = self.to_bytes(False)
        return sha256(sha256(payload).digest()).digest()[::-1].hex()

    def serialize(self):
        return self.to_bytes().hex()
