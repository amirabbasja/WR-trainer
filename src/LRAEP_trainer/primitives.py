import base64
from dataclasses import dataclass
from enum import IntEnum
import math
import struct

from .hashing import blake3, chunk_cv, parent_cv

MERKLE_LEAF_SIZE = 1024
PENALTY_BASE_RANK = 128
PUBLICDATA_SIZE = 164
MIN_MOE_PUBLICDATA_SIZE = 199
PUBLICDATA_MAX_SIZE = 199 + 128 * 4 + 1024 * 4


def _unsigned(value, bits, label):
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError(f'{label} must be an unsigned {bits}-bit integer')
    return value


def _bytes(value, length, label):
    if not isinstance(value, (bytes, bytearray, memoryview, list, tuple)):
        raise ValueError(f'{label} must be bytes')
    value = bytes(value)
    if len(value) != length:
        raise ValueError(f'{label} must contain {length} bytes')
    return value


@dataclass
class IncompleteBlockHeader:
    version: int
    prev_block: bytes
    merkle_root: bytes
    timestamp: int
    nbits: int
    SERIALIZED_SIZE = 76

    def __post_init__(self):
        for name in ('version', 'timestamp', 'nbits'):
            _unsigned(getattr(self, name), 32, name)
        self.prev_block = _bytes(self.prev_block, 32, 'prev_block')
        self.merkle_root = _bytes(self.merkle_root, 32, 'merkle_root')

    def to_bytes(self):
        self.__post_init__()
        return struct.pack('<I', self.version) + self.prev_block[::-1] + self.merkle_root[::-1] + struct.pack('<II', self.timestamp, self.nbits)

    @classmethod
    def from_bytes(cls, data):
        data = _bytes(data, cls.SERIALIZED_SIZE, 'header')
        return cls(struct.unpack('<I', data[:4])[0], data[4:36][::-1], data[36:68][::-1], *struct.unpack('<II', data[68:]))


class MMAType(IntEnum):
    Int7xInt7ToInt32 = 0

    @property
    def tensor_dtype(self):
        return 'int8'


@dataclass
class PeriodicPattern:
    shape: list
    NUM_DIMS = 3

    def __post_init__(self):
        self.shape = [tuple(pair) for pair in self.shape]
        if len(self.shape) != self.NUM_DIMS or any(len(pair) != 2 for pair in self.shape):
            raise ValueError('pattern requires exactly three stride/length pairs')
        for pair in self.shape:
            for value in pair:
                if _unsigned(value, 32, 'pattern value') == 0:
                    raise ValueError('pattern strides and lengths must be positive')

    @classmethod
    def from_bytes(cls, data):
        data = _bytes(data, 6, 'pattern')
        shape, minimum, done = [], 1, False
        for factor, length in zip(data[::2], data[1::2]):
            factor, length = factor + 1, length + 1
            if length == 1 or done:
                if factor != 1 or length != 1:
                    raise ValueError('noncanonical pattern')
                done = True
            elif factor <= 1 and minimum != 1:
                raise ValueError('a contiguous stride must not be split')
            if minimum * factor * length > 1 << 24:
                raise ValueError('pattern period exceeds 2^24')
            stride = minimum * factor
            shape.append((stride, length))
            minimum = stride * length
        return cls(shape)

    @classmethod
    def from_list(cls, pattern):
        pattern = list(pattern)
        if not pattern or pattern[0] != 0 or any(type(x) is not int or x < 0 for x in pattern) or any(a >= b for a, b in zip(pattern, pattern[1:])):
            raise ValueError('pattern must start at zero and be sorted and unique')
        shape = []
        while len(pattern) > 1:
            for period in range(1, len(pattern)):
                if len(pattern) % period == 0 and all(pattern[i] + pattern[period] == pattern[i + period] for i in range(len(pattern) - period)):
                    shape.append((pattern[period], len(pattern) // period))
                    pattern = pattern[:period]
                    break
            else:
                raise ValueError('indices do not form a periodic pattern')
        shape.reverse()
        if len(shape) > cls.NUM_DIMS:
            raise ValueError('pattern has too many dimensions')
        period = math.prod(shape[-1]) if shape else 1
        shape.extend([(period, 1)] * (cls.NUM_DIMS - len(shape)))
        result = cls(shape)
        if not result.is_valid():
            raise ValueError('pattern cannot be serialized canonically')
        return result

    def to_bytes(self):
        result, minimum = bytearray(), 1
        for stride, length in self.shape:
            if stride % minimum or not 1 <= stride // minimum <= 256 or not 1 <= length <= 256:
                raise ValueError('pattern stride/length cannot be encoded')
            result.extend((stride // minimum - 1, length - 1))
            minimum = stride * length
        return bytes(result)

    def to_list(self):
        result = [0]
        for stride, length in self.shape:
            result = [x + i * stride for i in range(length) for x in result]
        return result

    @property
    def period(self):
        return math.prod(self.shape[-1])

    @property
    def size(self):
        return math.prod(length for _, length in self.shape)

    def offset_is_valid(self, offset):
        if type(offset) is not int or offset < 0:
            return False
        for stride, length in reversed(self.shape):
            offset %= stride * length
            if offset >= stride:
                return False
        return True

    def indices_with_offset(self, offset):
        return [x + offset for x in self.to_list()]

    def is_valid(self):
        try:
            return self.from_bytes(self.to_bytes()).shape == self.shape
        except (ValueError, TypeError, OverflowError):
            return False


@dataclass
class MoEConfig:
    e: int
    top_k: int

    def __post_init__(self):
        _unsigned(self.e, 16, 'e')
        _unsigned(self.top_k, 16, 'top_k')
        if not 0 < self.top_k < self.e:
            raise ValueError('MoE requires 0 < top_k < e')


@dataclass
class MiningConfiguration:
    common_dim: int
    rank: int
    mma_type: MMAType
    rows_pattern: PeriodicPattern
    cols_pattern: PeriodicPattern
    moe: MoEConfig | None = None
    SERIALIZED_SIZE = 52

    def to_bytes(self):
        _unsigned(self.common_dim, 32, 'common_dim')
        _unsigned(self.rank, 16, 'rank')
        mma = MMAType(self.mma_type)
        trailer = bytes(32) if self.moe is None else struct.pack('<HH', self.moe.e, self.moe.top_k) + bytes(28)
        return struct.pack('<IHH', self.common_dim, self.rank, mma) + self.rows_pattern.to_bytes() + self.cols_pattern.to_bytes() + trailer

    @classmethod
    def from_bytes(cls, data):
        data = _bytes(data, cls.SERIALIZED_SIZE, 'mining configuration')
        k, rank, mma = struct.unpack('<IHH', data[:8])
        e, top_k = struct.unpack('<HH', data[20:24])
        if any(data[24:]) or (e == 0 and top_k):
            raise ValueError('noncanonical mining configuration trailer')
        return cls(k, rank, MMAType(mma), PeriodicPattern.from_bytes(data[8:14]), PeriodicPattern.from_bytes(data[14:20]), MoEConfig(e, top_k) if e else None)

    @property
    def hash_tile_h(self):
        return self.rows_pattern.size

    @property
    def hash_tile_w(self):
        return self.cols_pattern.size

    @property
    def rounded_common_dim(self):
        if self.rank <= 0:
            raise ValueError('rank must be positive')
        return self.common_dim - self.common_dim % self.rank


def penalized_target_bound(target, mining_config):
    _unsigned(target, 256, 'target')
    if mining_config.rank <= 0:
        return None
    factor = mining_config.hash_tile_h * mining_config.hash_tile_w * (mining_config.common_dim // mining_config.rank) * PENALTY_BASE_RANK
    result = target * factor
    return result if factor and result < 1 << 256 else None


def pad_to_chunk_boundary(data):
    data = bytes(data)
    return data + bytes(-len(data) % MERKLE_LEAF_SIZE)


class MerkleTree:
    def __init__(self, data, key):
        self.data = bytes(data)
        self.key = _bytes(key, 32, 'key')
        if not self.data:
            raise ValueError('Merkle data must not be empty')
        if len(self.data) <= MERKLE_LEAF_SIZE:
            self.layers = [[blake3(self.data, key=self.key).digest()]]
        else:
            leaves = [chunk_cv(self.data[offset:offset + MERKLE_LEAF_SIZE], self.key, offset // MERKLE_LEAF_SIZE) for offset in range(0, len(self.data), MERKLE_LEAF_SIZE)]
            self.layers = [leaves]
            while len(self.layers[-1]) > 1:
                layer = self.layers[-1]
                self.layers.append([parent_cv(layer[i], layer[i + 1], self.key, root=len(layer) == 2) if i + 1 < len(layer) else layer[i] for i in range(0, len(layer), 2)])

    @property
    def root(self):
        return self.layers[-1][0]

    @property
    def leaf_hashes(self):
        return list(self.layers[0])

    def get_multileaf_proof(self, leaf_indices):
        selected = sorted(set(leaf_indices))
        total = len(self.layers[0])
        if not selected or any(type(i) is not int or not 0 <= i < total for i in selected):
            raise ValueError('Merkle leaf indices are out of range')
        leaves = [self.data[i * MERKLE_LEAF_SIZE:(i + 1) * MERKLE_LEAF_SIZE].ljust(MERKLE_LEAF_SIZE, b'\0') for i in selected]
        current, siblings, width = set(selected), [], total
        for layer in self.layers[:-1]:
            for i in sorted(current):
                sibling = i ^ 1
                if sibling < width and sibling not in current:
                    siblings.append(layer[sibling])
            current = {i // 2 for i in current}
            width = (width + 1) // 2
        return MerkleProof(leaves, selected, self.root, siblings, total)

    @staticmethod
    def compute_leaf_indices_from_rows(row_indices, shape):
        if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError('matrix shape must contain two positive dimensions')
        indices = set()
        for row in row_indices:
            if type(row) is not int or not 0 <= row < shape[0]:
                raise ValueError('matrix row is out of range')
            indices.update(range(row * shape[1] // MERKLE_LEAF_SIZE, ((row + 1) * shape[1] - 1) // MERKLE_LEAF_SIZE + 1))
        return sorted(indices)


class MerkleProof:
    def __init__(self, leaf_data, leaf_indices, root, siblings, total_leaves):
        self.leaf_data = [_bytes(leaf, MERKLE_LEAF_SIZE, 'leaf') for leaf in leaf_data]
        self.leaf_indices = list(leaf_indices)
        self.root = _bytes(root, 32, 'root')
        self.siblings = [_bytes(sibling, 32, 'sibling') for sibling in siblings]
        self.total_leaves = total_leaves

    def sanity_check(self):
        if type(self.total_leaves) is not int or self.total_leaves <= 0:
            raise ValueError('Merkle tree requires a positive leaf count')
        if not self.leaf_indices or len(self.leaf_indices) != len(self.leaf_data) or self.leaf_indices != sorted(set(self.leaf_indices)) or any(type(i) is not int or not 0 <= i < self.total_leaves for i in self.leaf_indices):
            raise ValueError('Merkle leaf indices must be sorted, unique and in range')

    def compute_root(self, key):
        try:
            self.sanity_check()
            if self.total_leaves == 1:
                return blake3(self.leaf_data[0], key=key).digest() if not self.siblings and self.leaf_indices == [0] else None
            current = {i: chunk_cv(data, key, i) for i, data in zip(self.leaf_indices, self.leaf_data)}
            siblings, used, width = self.siblings, 0, self.total_leaves
            while width > 1:
                next_level = {}
                for i in sorted(current):
                    if i // 2 in next_level:
                        continue
                    sibling = i ^ 1
                    if sibling >= width:
                        next_level[i // 2] = current[i]
                        continue
                    if sibling in current:
                        other = current[sibling]
                    else:
                        if used >= len(siblings):
                            return None
                        other, used = siblings[used], used + 1
                    left, right = (current[i], other) if i % 2 == 0 else (other, current[i])
                    next_level[i // 2] = parent_cv(left, right, key, root=width == 2)
                current = next_level
                width = (width + 1) // 2
            return current.get(0) if used == len(siblings) else None
        except (ValueError, TypeError, IndexError, OverflowError):
            return None

    def verify(self, key):
        return self.compute_root(key) == self.root

    def extract_bytes(self, global_start, length):
        if type(global_start) is not int or type(length) is not int or global_start < 0 or length < 0:
            raise ValueError('invalid Merkle extraction range')
        chunks, end, position = dict(zip(self.leaf_indices, self.leaf_data)), global_start + length, global_start
        result = bytearray()
        while position < end:
            index, offset = divmod(position, MERKLE_LEAF_SIZE)
            if index not in chunks:
                raise ValueError('Merkle leaves do not cover required matrix bytes')
            size = min(MERKLE_LEAF_SIZE - offset, end - position)
            result.extend(chunks[index][offset:offset + size])
            position += size
        return bytes(result)


@dataclass
class MatrixMerkleProof:
    proof: MerkleProof
    row_indices: list

    def __post_init__(self):
        self.row_indices = list(self.row_indices)

    @property
    def root(self):
        return self.proof.root


class _Reader:
    def __init__(self, data):
        self.data, self.position = bytes(data), 0

    def read(self, size):
        if size < 0 or self.position + size > len(self.data):
            raise ValueError('truncated proof')
        start, self.position = self.position, self.position + size
        return self.data[start:self.position]

    def integer(self):
        return struct.unpack('<Q', self.read(8))[0]

    def vector(self, read, minimum_size):
        count = self.integer()
        if count > (len(self.data) - self.position) // minimum_size:
            raise ValueError('invalid proof vector length')
        return [read() for _ in range(count)]


def _pack_int(value):
    return struct.pack('<Q', _unsigned(value, 64, 'proof integer'))


def _pack_vec(values, pack):
    return _pack_int(len(values)) + b''.join(pack(value) for value in values)


def _serialize_matrix(matrix):
    proof = matrix.proof
    return _pack_vec(proof.leaf_data, lambda leaf: _pack_int(len(leaf)) + leaf) + _pack_vec(proof.leaf_indices, _pack_int) + _pack_int(proof.total_leaves) + proof.root + _pack_vec(proof.siblings, bytes) + _pack_vec(matrix.row_indices, _pack_int)


def _deserialize_matrix(reader):
    leaves = reader.vector(lambda: reader.read(reader.integer()), 8)
    indices = reader.vector(reader.integer, 8)
    total, root = reader.integer(), reader.read(32)
    siblings = reader.vector(lambda: reader.read(32), 32)
    rows = reader.vector(reader.integer, 8)
    return MatrixMerkleProof(MerkleProof(leaves, indices, root, siblings, total), rows)


class PlainProof:
    def __init__(self, m, n, k, noise_rank, a_merkle_proof, bt_merkle_proof, moe=None):
        self.m, self.n, self.k, self.noise_rank = m, n, k, noise_rank
        self.a, self.bt, self.moe = a_merkle_proof, bt_merkle_proof, moe

    @property
    def min_cert_version(self):
        return 2 if self.moe is not None else 1

    def to_base64(self):
        if self.moe is not None:
            raise ValueError('MoE plain proofs are unavailable')
        data = b''.join(_pack_int(value) for value in (self.m, self.n, self.k, self.noise_rank)) + _serialize_matrix(self.a) + _serialize_matrix(self.bt) + b'\0'
        return base64.b64encode(data).decode('ascii')

    @classmethod
    def from_base64(cls, data):
        if not isinstance(data, str) or len(data) > 24 << 20:
            raise ValueError('plain proof must be a bounded base64 string')
        try:
            reader = _Reader(base64.b64decode(data, validate=True))
            values = [reader.integer() for _ in range(4)]
            result = cls(*values, _deserialize_matrix(reader), _deserialize_matrix(reader))
            trailing = reader.data[reader.position:]
            if trailing not in (b'', b'\0'):
                raise ValueError('unsupported or trailing plain proof data')
            return result
        except (ValueError, TypeError, struct.error):
            raise ValueError('invalid plain proof encoding') from None


@dataclass
class ZKProof:
    public_data: bytes
    proof_data: bytes

    def __post_init__(self):
        self.public_data, self.proof_data = bytes(self.public_data), bytes(self.proof_data)


def nbits_to_difficulty(nbits):
    _unsigned(nbits, 32, 'nbits')
    exponent, mantissa = nbits >> 24, nbits & 0xFFFFFF
    if not exponent or not mantissa or mantissa & 0x800000:
        return 0
    target = mantissa << (8 * (exponent - 3)) if exponent > 3 else mantissa >> (8 * (3 - exponent))
    return target if target < 1 << 256 else 0


def check_cert_version_eligible(cert_version, plain_proof):
    if type(cert_version) is not int or cert_version not in (1, 2, 3):
        raise ValueError('unsupported certificate version')
    if plain_proof.min_cert_version > cert_version:
        raise ValueError('proof requires a later certificate version')


def _proof_digest(cert_version, header, proof):
    import numpy as np
    from .wr_work import _uniform_rows, _permutation
    check_cert_version_eligible(cert_version, proof)
    if proof.moe is not None:
        raise ValueError('MoE proofs are rejected by the dense-only rule')
    rank, k = proof.noise_rank, proof.k
    if rank != 128 or type(k) is not int or not 2048 <= k <= 65536 or k % 64:
        raise ValueError('unsupported proof rank or common dimension')
    for name, dimension in (('m', proof.m), ('n', proof.n)):
        if type(dimension) is not int or not 1 <= dimension <= 1 << 24:
            raise ValueError(f'invalid proof dimension {name}')
    patterns, signals = [], []
    for matrix, dimension in ((proof.a, proof.m), (proof.bt, proof.n)):
        rows = matrix.row_indices
        if not rows or rows != sorted(set(rows)) or any(type(i) is not int or not 0 <= i < dimension for i in rows):
            raise ValueError('invalid proof matrix rows')
        pattern = PeriodicPattern.from_list([i - rows[0] for i in rows])
        if not pattern.offset_is_valid(rows[0]):
            raise ValueError('invalid proof pattern offset')
        if matrix.proof.total_leaves != (dimension * k + MERKLE_LEAF_SIZE - 1) // MERKLE_LEAF_SIZE:
            raise ValueError('Merkle tree leaf count differs from declared dimensions')
        signal = np.stack([np.frombuffer(matrix.proof.extract_bytes(i * k, k), dtype=np.int8) for i in rows])
        if (signal < -64).any() or (signal > 64).any():
            raise ValueError('proof signal values must lie in [-64, 64]')
        patterns.append(pattern)
        signals.append(signal)
    h, w = (pattern.size for pattern in patterns)
    if h % 2 or w % 2 or not 32 <= h * w <= 256 or (h + w) * k > 1 << 22:
        raise ValueError('invalid proof tile shape')
    mining = MiningConfiguration(k, rank, MMAType.Int7xInt7ToInt32, *patterns)
    key = blake3(header.to_bytes() + mining.to_bytes()).digest()
    if not proof.a.proof.verify(key) or not proof.bt.proof.verify(key):
        raise ValueError('Merkle proof commitment does not match header and configuration')
    hash_a, hash_b = proof.a.root, proof.bt.root
    if cert_version == 3:
        salts = [blake3(b'pearl/cert-v3/noise-seed/' + label).digest() for label in (b'A', b'B')]
        hash_a = blake3(hash_a + struct.pack('<I', proof.m) + bytes(28), key=salts[0]).digest()
        hash_b = blake3(hash_b + struct.pack('<I', proof.n) + bytes(28), key=salts[1]).digest()
    seed_b = blake3(key + hash_b).digest()
    seed_a = blake3(seed_b + hash_a).digest()
    labels = (b'A_tensor' + bytes(24), b'B_tensor' + bytes(24))
    noised = []
    for matrix, signal, label, seed in zip((proof.a, proof.bt), signals, labels, (seed_a, seed_b)):
        uniform = _uniform_rows(matrix.row_indices, label, seed)
        first, second = _permutation(k, label, seed)
        noised.append((signal.astype(np.int16) + uniform[:, first] - uniform[:, second]).astype(np.int8).astype(np.int32))
    cumulative = np.zeros((h, w), dtype=np.int32)
    transcript = [0] * 16
    for reduction, offset in enumerate(range(0, k - k % rank, rank)):
        cumulative += noised[0][:, offset:offset + rank] @ noised[1][:, offset:offset + rank].T
        combined = int(np.bitwise_xor.reduce(cumulative.view(np.uint32).ravel()))
        slot = reduction % 16
        value = transcript[slot]
        transcript[slot] = (((value << 13) | (value >> 19)) & 0xFFFFFFFF) ^ combined
    return blake3(struct.pack('<16I', *transcript), key=seed_a).digest(), mining


def verify_plain_proof_for_cert_version(cert_version, block_header, plain_proof, nbits_override=None):
    try:
        digest, config = _proof_digest(cert_version, block_header, plain_proof)
        base = nbits_to_difficulty(block_header.nbits if nbits_override is None else nbits_override)
        factor = config.hash_tile_h * config.hash_tile_w * config.rounded_common_dim
        target = min((1 << 256) - 1, base * factor)
        penalty = min((1 << 256) - 1, base * config.hash_tile_h * config.hash_tile_w * (config.common_dim // config.rank) * PENALTY_BASE_RANK)
        if int.from_bytes(digest, 'little') > min(target, penalty):
            return False, 'proof digest does not meet difficulty target'
        return True, 'Mining solution verified successfully'
    except (ValueError, TypeError, AttributeError, OverflowError, struct.error) as error:
        return False, str(error)


def verify_plain_proof_v1(block_header, plain_proof, nbits_override=None):
    return verify_plain_proof_for_cert_version(1, block_header, plain_proof, nbits_override)


def verify_plain_proof_v2(block_header, plain_proof, nbits_override=None):
    return verify_plain_proof_for_cert_version(2, block_header, plain_proof, nbits_override)


def verify_plain_proof_v3(block_header, plain_proof, nbits_override=None):
    return verify_plain_proof_for_cert_version(3, block_header, plain_proof, nbits_override)


def generate_proof_for_cert_version(cert_version, block_header, plain_proof):
    check_cert_version_eligible(cert_version, plain_proof)
    raise RuntimeError('Local ZK certificate generation is unavailable; WR and plain pool shares do not require it')


def verify_proof_for_cert_version(cert_version, block_header, proof):
    if cert_version not in (1, 2, 3):
        raise ValueError('unsupported certificate version')
    raise RuntimeError('Local ZK certificate verification is unavailable')
