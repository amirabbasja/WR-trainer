from dataclasses import dataclass, field
from hashlib import sha256


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
        opcodes = {'OP_0': 0, 'OP_FALSE': 0, 'OP_1NEGATE': 0x4F, 'OP_RETURN': 0x6A, 'OP_TRUE': 0x51}
        opcodes.update({f'OP_{i}': 0x50 + i for i in range(1, 17)})
        for value in self.script:
            if type(value) is int:
                result.extend(_script_number(value))
            elif isinstance(value, str) and value in opcodes:
                result.append(opcodes[value])
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
class TxInput:
    txid: str
    txout_index: int
    script_sig: Script = field(default_factory=Script)
    sequence: bytes = b'\xff\xff\xff\xff'

    def to_bytes(self):
        txid = bytes.fromhex(self.txid)
        if len(txid) != 32 or type(self.txout_index) is not int or not 0 <= self.txout_index <= 0xFFFFFFFF or len(self.sequence) != 4:
            raise ValueError('Invalid transaction input')
        script = self.script_sig.to_bytes()
        return txid[::-1] + self.txout_index.to_bytes(4, 'little') + encode_varint(len(script)) + script + self.sequence


@dataclass
class TxOutput:
    amount: int
    script_pubkey: Script

    def to_bytes(self):
        if type(self.amount) is not int or not 0 <= self.amount <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError('Invalid transaction output amount')
        script = self.script_pubkey.to_bytes()
        return self.amount.to_bytes(8, 'little') + encode_varint(len(script)) + script


@dataclass
class TxWitnessInput:
    stack: list[str]

    def to_bytes(self):
        items = [bytes.fromhex(item) for item in self.stack]
        return encode_varint(len(items)) + b''.join(encode_varint(len(item)) + item for item in items)


@dataclass
class Transaction:
    inputs: list[TxInput]
    outputs: list[TxOutput]
    locktime: bytes = bytes(4)
    version: bytes = b'\x01\x00\x00\x00'
    has_segwit: bool = False
    witnesses: list[TxWitnessInput] | None = None

    def to_bytes(self, has_segwit=None):
        if len(self.version) != 4 or len(self.locktime) != 4 or not self.inputs or not self.outputs:
            raise ValueError('Invalid transaction header or empty inputs/outputs')
        witness = self.has_segwit if has_segwit is None else bool(has_segwit and self.has_segwit)
        result = self.version + (b'\x00\x01' if witness else b'')
        result += encode_varint(len(self.inputs)) + b''.join(item.to_bytes() for item in self.inputs)
        result += encode_varint(len(self.outputs)) + b''.join(item.to_bytes() for item in self.outputs)
        if witness:
            if self.witnesses is None or len(self.witnesses) != len(self.inputs):
                raise ValueError('Witnesses must match transaction inputs')
            result += b''.join(item.to_bytes() for item in self.witnesses)
        return result + self.locktime

    def get_txid(self):
        payload = self.to_bytes(False)
        return sha256(sha256(payload).digest()).digest()[::-1].hex()

    def serialize(self):
        return self.to_bytes().hex()
