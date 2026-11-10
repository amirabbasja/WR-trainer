"""Training record payloads with canonical byte serialization."""

from dataclasses import dataclass, field
from hashlib import sha256


_OPERATIONS = {
    'training.zero': 0x00,
    'training.false': 0x00,
    'training.negative_one': 0x4F,
    'training.attestation': 0x6A,
    'training.true': 0x51,
}
_OPERATIONS.update({f'training.value_{i}': 0x50 + i for i in range(1, 17)})
_OPTIONAL_SECTION = b'\x00\x01'

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
    raise ValueError('Training payload element is too large')


def _payload_integer(value):
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


class TrainingPayload:
    def __init__(self, payload=None):
        self.payload = list(payload or [])
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
        for value in self.payload:
            if type(value) is int:
                result.extend(_payload_integer(value))
            elif isinstance(value, str) and value in _OPERATIONS:
                result.append(_OPERATIONS[value])
            elif isinstance(value, str):
                result.extend(_push(bytes.fromhex(value)))
            elif isinstance(value, bytes):
                result.extend(_push(value))
            else:
                raise ValueError('Invalid payload element')
        return bytes(result)

    def to_hex(self):
        return self.to_bytes().hex()


@dataclass
class TrainingInput:
    record_id: str
    source_output_index: int
    input_payload: TrainingPayload = field(default_factory=TrainingPayload)
    sequence: bytes = b'\xff\xff\xff\xff'

    def to_bytes(self):
        record_id = bytes.fromhex(self.record_id)
        if len(record_id) != 32 or type(self.source_output_index) is not int or not 0 <= self.source_output_index <= 0xFFFFFFFF or len(self.sequence) != 4:
            raise ValueError('Invalid training record input')
        payload = self.input_payload.to_bytes()
        return record_id[::-1] + self.source_output_index.to_bytes(4, 'little') + encode_varint(len(payload)) + payload + self.sequence


@dataclass
class TrainingOutput:
    amount: int
    output_payload: TrainingPayload

    def to_bytes(self):
        if type(self.amount) is not int or not 0 <= self.amount <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError('Invalid training record output amount')
        payload = self.output_payload.to_bytes()
        return self.amount.to_bytes(8, 'little') + encode_varint(len(payload)) + payload


@dataclass
class TrainingAttestation:
    stack: list[str]

    def to_bytes(self):
        items = [bytes.fromhex(item) for item in self.stack]
        return encode_varint(len(items)) + b''.join(encode_varint(len(item)) + item for item in items)


@dataclass
class TrainingRecord:
    inputs: list[TrainingInput]
    outputs: list[TrainingOutput]
    release_step: bytes = bytes(4)
    version: bytes = b'\x01\x00\x00\x00'
    tagged: bool = False
    attestations: list[TrainingAttestation] | None = None

    def to_bytes(self, tagged=None):
        if len(self.version) != 4 or len(self.release_step) != 4 or not self.inputs or not self.outputs:
            raise ValueError('Invalid training record header or empty inputs/outputs')
        attestation = self.tagged if tagged is None else bool(tagged and self.tagged)
        result = self.version + (_OPTIONAL_SECTION if attestation else b'')
        result += encode_varint(len(self.inputs)) + b''.join(item.to_bytes() for item in self.inputs)
        result += encode_varint(len(self.outputs)) + b''.join(item.to_bytes() for item in self.outputs)
        if attestation:
            if self.attestations is None or len(self.attestations) != len(self.inputs):
                raise ValueError('Attestations must match training record inputs')
            result += b''.join(item.to_bytes() for item in self.attestations)
        return result + self.release_step

    def get_record_id(self):
        payload = self.to_bytes(False)
        return sha256(sha256(payload).digest()).digest()[::-1].hex()

    def serialize(self):
        return self.to_bytes().hex()
