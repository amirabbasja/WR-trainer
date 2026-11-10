import torch
from workrunner.stream_digest import StreamDigest
from job_gateway.comm.training_types import CommitmentHash
from workrunner.protocol_types import S1Configuration
from workrunner.noise_bounds import certificate_salt
from workrunner.fixed_constants import FixedAlias, constant_buffer
from .matrix_tree import MatrixTree
def seed_salt(label):
    return certificate_salt(StreamDigest, label)



def _bind_root(tree_root: bytes, dim: int, salt: bytes) -> bytes:
    return StreamDigest(tree_root + dim.to_bytes(4, 'little') + bytes(28), key=salt).digest()

def bind_root_a(a_tree_root: bytes, m: int) -> bytes:
    with constant_buffer(FixedAlias('training.tensor.left_operand_label')) as label:
        return _bind_root(a_tree_root, m, seed_salt(label))

def bind_root_b(b_tree_root: bytes, n: int) -> bytes:
    with constant_buffer(FixedAlias('training.tensor.right_operand_label')) as label:
        return _bind_root(b_tree_root, n, seed_salt(label))

class CommitmentHasher:

    @staticmethod
    def get_key(incomplete_header_bytes: bytes, S1_config: S1Configuration) -> bytes:
        return StreamDigest(incomplete_header_bytes + S1_config.to_bytes()).digest()

    @classmethod
    def commitment_hash(cls, A: torch.Tensor, B: torch.Tensor, incomplete_header_bytes: bytes, S1_config: S1Configuration, *, salted_dims: tuple[int, int] | None) -> CommitmentHash:
        key = cls.get_key(incomplete_header_bytes, S1_config)
        tree_a = MatrixTree(A, key)
        tree_b = MatrixTree(B.T, key)
        return cls.commitment_hash_from_tree_roots(tree_a.root, tree_b.root, key, salted_dims=salted_dims)

    @staticmethod
    def get_commitment_B_key(key: bytes, B_tree_root: bytes) -> bytes:
        return StreamDigest(key + B_tree_root).digest()

    @staticmethod
    def get_commitment_A_key(commitment_B: bytes, A_tree_root: bytes) -> bytes:
        return StreamDigest(commitment_B + A_tree_root).digest()

    @staticmethod
    def get_offsets_hash(routing_offsets: list[int], key: bytes) -> bytes:
        offsets = torch.tensor(routing_offsets, dtype=torch.int32)
        return MatrixTree.tensor_hash(offsets, key)

    @staticmethod
    def combine_routing_tree_roots(hash_a: bytes, routing_root: bytes, offsets_root: bytes) -> bytes:
        hash_routing = StreamDigest(routing_root + offsets_root).digest()
        return StreamDigest(hash_a + hash_routing).digest()

    @classmethod
    def commitment_hash_from_tree_roots(cls, A_tree_root: bytes, B_tree_root: bytes, key: bytes, *, routing_root: bytes | None=None, offsets_root: bytes | None=None, salted_dims: tuple[int, int] | None) -> CommitmentHash:
        if salted_dims is not None:
            m, n = salted_dims
            A_tree_root = bind_root_a(A_tree_root, m)
            B_tree_root = bind_root_b(B_tree_root, n)
        moe_args = (routing_root, offsets_root)
        if any((arg is not None for arg in moe_args)):
            if any((arg is None for arg in moe_args)):
                raise ValueError('routing_root and offsets_root must be provided together')
            A_tree_root = cls.combine_routing_tree_roots(A_tree_root, routing_root, offsets_root)
        commitment_B = cls.get_commitment_B_key(key, B_tree_root)
        commitment_A = cls.get_commitment_A_key(commitment_B, A_tree_root)
        return CommitmentHash(commitment_A, commitment_B)
