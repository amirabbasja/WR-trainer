from __future__ import annotations
from collections.abc import Iterable
from .fixed_constants import FixedAlias, constant_text
from .training_constants import local_seed_field
from . import training_constants as wire

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
    local_seed_hex = _seed(local_seed, 'local_seed')
    device_seed_hex = _seed(device_seed, 'device_seed')
    row_label = constant_text(FixedAlias('training.field.input_row_index_label'))
    column_label = constant_text(FixedAlias('training.field.weight_column_index_label'))
    baseline = constant_text(FixedAlias('training.field.signal_baseline'))
    tile_height = constant_text(FixedAlias('training.field.matrix_tile_height'))
    tile_width = constant_text(FixedAlias('training.field.matrix_tile_width'))
    params = {
        constant_text(FixedAlias('training.field.input_row_indices')): [
            _i32(value, row_label) for value in a_rows
        ],
        constant_text(FixedAlias('training.field.weight_column_indices')): [
            _i32(value, column_label) for value in b_cols
        ],
        'k': _i32(k, 'k'),
        'm': _i32(m, 'm'),
        local_seed_field(): local_seed_hex,
        'n': _i32(n, 'n'),
        wire.SEED: local_seed_hex,
        'rank': _i32(rank, 'rank'),
        baseline: _i32(sb, baseline),
        constant_text(FixedAlias('training.field.signal_baseline_parameters')): [],
        constant_text(FixedAlias('training.field.tensor_tail_mode')): 1,
        tile_height: _i32(tile_h, tile_height),
        tile_width: _i32(tile_w, tile_width),
        wire.DEVICE_SEED: device_seed_hex,
    }
    return {'id': request_id, 'method': wire.SEND_RESULT, 'params': [params]}
