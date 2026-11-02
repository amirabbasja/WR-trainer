# Conversation: Proof-of-work constants

Date: 2026-11-03 (Asia/Tehran)

## User

tell me all pow (proof of work) constants in this miner.

## Assistant — progress

I’ll trace the miner’s proof-of-work code and list its constants, their values, and where they’re used.

The PoW uses noisy matrix multiplication and a 256-bit digest. I found fixed algorithm values as well as configurable matrix sizes; I’m checking the CPU verifier and CUDA code for inline constants too.

The main values agree across the implementations: noise rank 128, a 16-word transcript, and a 13-bit transcript rotation. WR also fixes the signal fill at 64 and uses two 8-byte nonces. The target comes from each job, and the mining loop scales it by the configured work size.

## Assistant — answer

The miner uses **noisy matrix multiplication followed by keyed BLAKE3**, named `CNN_3ekalb` in the code. These are the PoW-related constants in this checkout, including inline values and configurable defaults.

**Core PoW values**

| Constant / parameter | Value |
|---|---|
| `RANK`, noise rank | **128** |
| `PENALTY_BASE_RANK` | **128** |
| `TRANSCRIPT_SIZE_U32` | **16 × uint32 = 64 bytes** |
| `HASH_ACCUMULATE_ROTATION` | **Left rotate by 13 bits**, equivalent to right rotate by 19 |
| `_ROTATE_RIGHT_MASK` | `8191` / `0x1FFF` |
| Transcript initial state | 16 zero words |
| Transcript slot | Slab number modulo **16** |
| Common-dimension slab size | **128** elements |
| Matrix storage / accumulation | `int8` / `int32` |
| `MMAType.Int7xInt7ToInt32` | Enum value **0** |
| `MERKLE_LEAF_SIZE` | **1024 bytes** |
| Merkle roots / hash keys | **32 bytes** |
| `POW_TARGET_HARDEST` | **0** |
| `POW_TARGET_EASIEST` | **2²⁵⁶ − 1** |
| Acceptance rule | Little-endian digest integer **≤ target** |

Sources: [GPU transcript implementation](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/gpu.py:5), [PoW helper constants](/workspace/LRAEP-WR-Trainer/src/LRAEP_helpers/python/S5_base/noisy_gemm.py:11), [primitives](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/primitives.py:11).

**Matrix defaults and supported limits**

These sizes are configurable; rank remains fixed at 128.

| Parameter | Default | Miner constraint |
|---|---:|---|
| `m` | **256** | 128–4096 |
| `n` | **256** | 128–4096 |
| `k` | **2048** | 2048–65536, multiple of 128 |
| `tile_h` | **8** | Even, 2–128 |
| `tile_w` | **8** | Even, 2–128 |
| Tile area | **64** | 32–256 cells |
| Selected matrix input size | — | `(tile_h + tile_w) × k ≤ 2²²` bytes |
| Divisibility | — | `m % tile_h == 0`, `n % tile_w == 0` |

The native CUDA path supports **8×8 hash tiles**. The bundled `NoisyGemm` helper independently defaults to **16×16 hash tiles** and **128×128 matmul tiles**, but the miner supplies its configured hash tile sizes.

Sources: [training configuration](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/protocol.py:29), [configuration validation](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/config.py:348), [helper defaults](/workspace/LRAEP-WR-Trainer/src/LRAEP_helpers/python/S5_base/noisy_gemm.py:56).

**Noise generation**

| Constant / fixed rule | Value |
|---|---|
| `noise_range` | **128** |
| Sparse entries per line | **2**, with coefficients **+1** and **−1** |
| Uniform factor samples | **−32 through 31** |
| Uniform byte mask | **63** / `0x3F` |
| Uniform zero-point subtraction | **32** |
| Rank index mask | **127** / `0x7F` |
| A domain label | `b'A_tensor' + bytes(24)` |
| B domain label | `b'B_tensor' + bytes(24)` |
| Random-hash counter preamble | **32 bytes**, eight 32-bit words |
| Counter numbering | Starts at **1** |
| Counter word | Word **0** for uniform factors; word **1** for sparse indices |
| Sparse random input | **4 bytes** per pair; **8 pairs** per digest |
| V3 noise salt prefix | `b'pearl/cert-v3/noise-seed/'` |
| V3 salt labels | `b'A'`, `b'B'` |
| V3 dimension binding | Root + **4-byte little-endian dimension** + **28 zero bytes** |

Sparse indices are calculated as:

```text
first  = value & 127
second = first XOR (1 + high32(127 × value))
```

Noise addition wraps to `int8`.

Sources: [noise generator](/workspace/LRAEP-WR-Trainer/src/LRAEP_helpers/python/S5_base/noise_generation.py:21), [noise labels](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/noise.py:4), [salt prefix](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/wire_constants.py:8), [root binding](/workspace/LRAEP-WR-Trainer/src/LRAEP_helpers/python/S5_base/commitment_hash.py:7).

**WR-specific work constants**

| Parameter | Value |
|---|---|
| Required certificate generation | **3** |
| `signal_base()` | **64** / `0x40` |
| Miner nonce | **8 bytes** |
| Worker extranonce | **8 bytes** |
| Nonce tail | First **16 digest bytes**, mapped using `byte % 129 − 64` |
| Tail value range | **−64 through 64** |
| Extra A rows | **1**; commitment uses `m + 1` |
| Tail placement | Last **16 bytes** of the extra A row |
| `sb` | **1** |
| `tail_mode` | **1** |
| `sb_p` | Empty list |
| Notify payload size | **200 bytes** |
| Decoded incomplete header | **76 bytes** |
| Received target | **32 bytes**, little-endian |
| Legacy protocol alias | `"pearlhash"` |

For the KEX path, random signal generation uses **−64 through 63**; the plain-proof verifier permits **−64 through 64**.

Sources: [WR work construction](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/wr_work.py:10), [WR protocol](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/wr_protocol.py:39), [submission fields](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/S6_requests.py:24), [KEX signals](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/protocol.py:103).

WR also hard-codes these decoding masks in [wr_protocol.py](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/wr_protocol.py:12):

```text
header_mask =
9d2f73c411a85be0367ad14e920cbf65f32884da176eb941cc530fa67de21974

work_prefix_mask =
4dc0d47dff723cf8f75a583b596cf848f5106dd0d0aa7bdded9d4d566a929ffaabea8985cf34a26af1b118e2c697f283f0caadd643b0975390a17c79aae32e69cf166b57a6f64dd02914e7e4
```

**Target calculation**

The target varies by job. The active mining loop, including WR, applies:

```text
final_target = job.target × tile_h × tile_w × floor(k / rank) × 128
```

With rank 128 and valid miner dimensions, this becomes:

```text
final_target = job.target × tile_h × tile_w × k
```

At the defaults, the multiplier is **131072 = 2¹⁷**. Overflow beyond 256 bits is rejected.

Compact `nbits` decoding uses an exponent shift of **24**, mantissa mask **`0x007FFFFF`**, sign bit **`0x00800000`**, and exponent bias **3**. The KEX local proof check additionally uses **`nbits_override=545259519` / `0x207FFFFF`** after checking the exact share target.

Sources: [active target selection](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/S3_training.py:299), [target multiplier](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/primitives.py:218), [KEX verification override](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/protocol.py:202).

**BLAKE3 constants**

| Parameter | Value |
|---|---|
| Digest / key / chaining-value size | **32 bytes** |
| Compression block | **64 bytes** |
| Chunk | **1024 bytes** |
| Word / counter widths | **32 / 64 bits** |
| Compression rounds | **7** |
| Mixing rotations | Right **16, 12, 8, 7** |
| Word mask | `0xFFFFFFFF` |
| Flags | `CHUNK_START=1`, `CHUNK_END=2`, `PARENT=4`, `ROOT=8`, `KEYED_HASH=16` |
| Key-derivation flags | Context **32**, material **64** |
| Single-block keyed-root flags | **27** = `16 | 8 | 2 | 1` |

```text
IV, hexadecimal:
6A09E667 BB67AE85 3C6EF372 A54FF53A
510E527F 9B05688C 1F83D9AB 5BE0CD19

Message permutation:
2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
```

Sources: [Python hash implementation](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/CNN_3ekalb.py:15), [native hash constants](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/CNN_3ekalb_native.cpp:8), [CUDA keyed hash](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/cuda/blake3.cuh:59).

**Proof and serialization constants**

| Constant | Value |
|---|---|
| `IncompleteBlockHeader.SERIALIZED_SIZE` | **76 bytes** |
| `S1Configuration.SERIALIZED_SIZE` | **52 bytes** |
| `PeriodicPattern.NUM_DIMS` | **3** |
| Encoded periodic pattern | **6 bytes**; factors/lengths 1–256; maximum period **2²⁴** |
| `PUBLICDATA_SIZE` | **164 bytes** |
| `MIN_MOE_PUBLICDATA_SIZE` | **199 bytes** |
| `PUBLICDATA_MAX_SIZE` | **4807 bytes** = `199 + 128×4 + 1024×4` |
| `PROOF_COMMITMENT_SIZE` | **32 bytes** |
| Complete header size | **108 bytes** |
| Certificate versions | Dense **1**, MoE **2**, V3 **3** |
| Certificate version / proof-length fields | **4 bytes** each |
| `ZK_MAX_PROOF_DATA_SIZE` | **60000 bytes** |
| `MAX_DECODED_PROOF_BYTES` | **4 MiB** |
| Plain-proof integer/vector-length fields | **8 bytes**, little-endian |
| Plain-proof Base64 parser limit | **24 MiB** of encoded input |

MoE constants are present, while this miner’s proof path enforces dense work.

Sources: [proof primitives](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/primitives.py:11), [certificate constants](/workspace/LRAEP-WR-Trainer/src/LRAEP_helpers/python/LRAEP_gateway/blockchain_utils/zk_certificate.py:10), [header commitment](/workspace/LRAEP-WR-Trainer/src/LRAEP_helpers/python/LRAEP_gateway/blockchain_utils/LRAEP_header.py:3), [submission proof limit](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/S3_protocol.py:18).

Performance settings also include a **256 MiB transcript scratch budget**, a maximum **16 slabs per chunk**, and an initial CUDA candidate capacity of **64**. These control execution and buffering without changing the PoW result. [Scratch settings](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/gpu.py:7), [candidate buffering](/workspace/LRAEP-WR-Trainer/src/LRAEP_trainer/native_cuda.py:174).

## User

save this conversation to an md file

## Assistant — progress

I’ll save the conversation as a Markdown file in the project folder.
