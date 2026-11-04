from __future__ import annotations
from collections.abc import Iterable
from .fixed_constants import constant_text
from .wire_constants import local_seed_field
from . import wire_constants as wire

def _i32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if not -(1 << 31) <= value < 1 << 31:
        raise ValueError(f'{name} is outside the signed 32-bit serializer range')
    return value

def _seed(value: bytes | str, name: str) -> str:
    if isinstance(value, str):
        if len(value) != 16:
            raise ValueError(f'{name} must be exactly 16 hex characters')
        try:
            value = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f'{name} must be hex') from exc
    if not isinstance(value, bytes) or len(value) != 8:
        raise ValueError(f'{name} must be exactly eight bytes')
    return value.hex()

def build_submission_request(request_id: int, *, local_seed: bytes | str, device_seed: bytes | str, m: int, n: int, k: int, rank: int, tile_h: int, tile_w: int, sb: int, a_rows: Iterable[int], b_cols: Iterable[int]) -> dict:
    if isinstance(request_id, bool) or not isinstance(request_id, int) or (not 0 <= request_id < 1 << 64):
        raise ValueError('request_id must be an unsigned 64-bit integer')
    S5 = _seed(local_seed, wire.LOCAL_SEED)
    worker = _seed(device_seed, wire.DEVICE_SEED)
    params = {constant_text(b'#`x99(9{'): [_i32(value, constant_text(b'#`x99(A0;{$-~0m')) for value in a_rows], constant_text(b'#rVX?%hU'): [_i32(value, constant_text(b'#rVX?%hZR?$-~0m')) for value in b_cols], 'k': _i32(k, 'k'), 'm': _i32(m, 'm'), local_seed_field(): S5, 'n': _i32(n, 'n'), wire.SEED: S5, 'rank': _i32(rank, 'rank'), constant_text(b')Wr'): _i32(sb, constant_text(b')Wr')), constant_text(b')W!JM'): [], constant_text(b')5gxr_{zw`!T'): 1, constant_text(b')6UDn_|E'): _i32(tile_h, constant_text(b')6UDn_|E')), constant_text(b')6UDn_|O'): _i32(tile_w, constant_text(b')6UDn_|O')), wire.DEVICE_SEED: worker}
    return {'id': request_id, 'method': wire.SEND_RESULT, 'params': [params]}
