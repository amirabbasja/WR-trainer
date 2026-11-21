# WR Trainer

The WR client follows the PearlHash exchange captured from WildRig Multi 0.52.3.
It starts each connection with login request ID `0` and defaults to the agent
`WildRig_0.52.3`. An explicitly configured custom agent still takes precedence.

The first `report_info` is due 15 seconds after login. Subsequent reports are due
120.6 seconds apart. The trainer sends a due report after completing a batch.

## VPN recovery

If the VPN health check or pool connection fails, hashing pauses and the failed
session closes. The trainer retries after `S3.reconnect_delay` seconds (with a
minimum delay of 0.1 seconds), including during the same scheduled interval.
It restarts a stopped VPN core, checks the configured `vpn.check_url` through the
VPN, and reconnects to the pool for fresh work before hashing resumes. With
`vpn.check_url` unset, the VPN core and pool connection establish readiness.
The notebook's live row shows `VPN: Disconnected; retrying in ...`,
`VPN: Disconnected; retrying`, and `VPN: Connected (...)` after recovery.
Interrupting the notebook stops retries. Restart the kernel to load this update.

## Telegram alerts

Set `telegram.enabled`, `telegram.tgBot_key`, and `telegram.tgChatID` in your
config to receive training error and VPN connection-loss alerts. Error alerts
include the trainer name, exception type, and message, and use a direct connection
after the training VPN closes. Job start/finish events and periodic progress
reports are not sent. The legacy `training_stats_tgBot_key` setting is accepted
but unused; `update_interval_seconds` controls retries for failed alerts.

## Native CUDA performance

Large 8×8 and 8×16 workloads use vectorized DP4A kernels with 128×64 blocks.
Two lanes share an 8×8 transcript tile; four share an 8×16 tile. Partial blocks
and matrices aligned only to four bytes retain safe fallback handling.
The same optimization is available in ANN-Trainer’s shared runtime. Native WR batches
reuse the header-dependent right operand across nonces and rebuild it when the
header changes. A cached tree path recomputes the left commitment from the
16-byte nonce tail; the left commitment and noise still change with each nonce.
Both contiguous 8×8 and captured 8×16 WR shapes use the preparation cache.
A header change or session cleanup releases the cached operands and tree path.
The notebook and CLI select this path automatically; restart a running worker
to load changed Python code and rebuild the CUDA library. Quiet runs can build
the library temporarily without leaving build artifacts. A compatible Linux
CUDA host and `nvcc` are still required for a fresh build.

Benchmark complete batches offline, including preparation and candidate
filtering, without reading your config or connecting to a pool:

```bash
python tools/benchmark_miner.py --iterations 100
python tools/benchmark_miner.py --iterations 100 --uncached --variant 7  # previous large 8×8 path
```

The default shape is 4096×4096×16384 with 8×8 tiles. Compilation and three
warmup batches are excluded; 20 measured batches use the runtime's `m*n*k`
rate accounting. Use `--iterations 100` for a longer run or `--stages` to
inspect costs (stage instrumentation adds synchronization). Offline rates
exclude pool latency, stale jobs, scheduling pauses, and CPU share verification.
Compare runs on the same idle GPU; concurrent work changes the measurement.

On the GTX 1660 SUPER used for validation, 100 complete offline batches improved
from 4.72 TH/s to 6.22 TH/s in WR-Trainer (Production measured 6.23 TH/s). [Benchmark results](tools/benchmarks/gtx1660-super.json)
record both checkouts and the comparison commands.

Run native build, GPU transcript, and preparation checks with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_native*.py'
```

## Production compute binaries

Production contains compiled CPU, CUDA, and benchmark libraries. Its C++ and
CUDA sources are maintained here under `native/production/` so the production
checkout can run without source files or compilers. Rebuild and verify them
from this WR-trainer checkout:

```bash
python tools/build_production_native.py
python tools/build_production_native.py --check
```

The build defaults to the sibling `Production-LREAP`; use `--production PATH`
for another checkout. It builds every supported `nvcc` GPU target by default;
use `--architectures 75 86` to select targets. The installed
`compute_manifest.json` records the targets, compiler version, source
fingerprints, and library hashes. The current Linux x86_64 build includes
SM 75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, and 121. Rebuild with an
appropriate toolkit in the trainer checkout for other platforms or targets.
Restart production workers after deploying new libraries.

<!-- native-word-codec:start -->
## Native word codec

ANN-Trainer's `native/word_codec.cpp` defines the exact aliases used
by both trainers and production. Training-related names identify internal constants. The
codec resolves them to the original text or bytes wherever the peer protocol,
proof format, diagnostics, or regression fixtures require those values.
Each C++ table row contains exactly two fields: the training alias and its
standard padded Base64 value. On an alias lookup, C++ decodes that Base64 inside
native memory and returns the original text or binary bytes.
Encoding compares the original bytes against a temporary decoded value and
returns its alias. The Python API and the peer's wire format stay unchanged.
Base64 is reversible encoding; the original values are documented below.

The native wire encoder/decoder also translates only the selected JSON fields
and methods and handles the peer's frame format. It uses the same encoded table;
ordinary message strings are preserved.

```python
from LRAEP_trainer.word_codec import FixedAlias, encode_fixed, fixed_buffer, transform_word

transform_word("legacy_protocol", "decode")
transform_word("pearlhash", "encode")
assert encode_fixed(b"target") == FixedAlias("training.field.acceptance_bound")
with fixed_buffer(FixedAlias("training.field.acceptance_bound")) as temporary:
    consume(temporary)
```

The corresponding standalone import is `workrunner.native_text_codec`.
`decode_word` and `encode_word` match structural protocol aliases exactly and
pass unlisted text through unchanged. `encode_fixed` and `fixed_buffer` with
a `FixedAlias` perform exact lookup across the full table. An unknown fixed
alias or value raises `ValueError`. Fixed values can include arbitrary binary
bytes. The codec performs no substring replacement.

### Placeholder table (decode direction)

| Training alias | Decoded value | Base64 encoded value (preview) |
| --- | --- | --- |
| `account` | `wallet` | `d2FsbGV0` |
| `legacy_protocol` | `pearlhash` | `cGVhcmxoYXNo` |
| `training_rate` | `hashrate` | `aGFzaHJhdGU=` |
| `server_seed` | `extranonce` | `ZXh0cmFub25j…` |
| `local_seed` | `miner_nonce` | `bWluZXJfbm9u…` |
| `device_seed` | `worker_nonce` | `d29ya2VyX25v…` |
| `training.subscribe` | `mining.subscribe` | `bWluaW5nLnN1…` |
| `training.authorize` | `mining.authorize` | `bWluaW5nLmF1…` |
| `training.submit` | `mining.submit` | `bWluaW5nLnN1…` |
| `training.notify` | `mining.notify` | `bWluaW5nLm5v…` |
| `training.reconnect` | `mining.reconnect` | `bWluaW5nLnJl…` |
| `noise_seed_domain` | `pearl/cert-v3/noise-seed/` | `cGVhcmwvY2Vy…` |
| `identity_prefix` | `prl1` | `cHJsMQ==` |
| `identity_mode_prefix` | `solo:` | `c29sbzo=` |
| `device_default` | `rig-01` | `cmlnLTAx` |
| `record_id` | `txid` | `dHhpZA==` |
| `parent_digest` | `previousblockhash` | `cHJldmlvdXNi…` |
| `size_limit` | `vsizelimit` | `dnNpemVsaW1p…` |
| `records` | `transactions` | `dHJhbnNhY3Rp…` |
| `poll_id` | `longpollid` | `bG9uZ3BvbGxp…` |
| `counter_range` | `noncerange` | `bm9uY2VyYW5n…` |
| `reward_metadata` | `coinbaseaux` | `Y29pbmJhc2Vh…` |
| `reward_amount` | `coinbasevalue` | `Y29pbmJhc2V2…` |
| `default_attestation` | `default_witness_commitment` | `ZGVmYXVsdF93…` |
| `required_version` | `requiredcertversion` | `cmVxdWlyZWRj…` |
| `template_request` | `getblocktemplate` | `Z2V0YmxvY2t0…` |
| `tip_request` | `getbestblockhash` | `Z2V0YmVzdGJs…` |
| `submit_record` | `submitblock` | `c3VibWl0Ymxv…` |
| `training.format.batch_frame_coefficients` | `1e62c4909baa47273a9650ad471b61ac63360c3c590e48592245bdbb2a05010408030263e81ef40609040bab3763667b` (48 bytes, hex) | `HmLEkJuqRyc6…` |
| `training.capability.append_attestation_record` | `coinbase/append` | `Y29pbmJhc2Uv…` |
| `training.capability.separate_attestation_payload` | `segwit` | `c2Vnd2l0` |
| `training.config.coordinator_section` | `S3` | `UzM=` |
| `training.config.legacy_batch_protocol` | `S7` | `Uzc=` |
| `training.field.acceptance_bound` | `target` | `dGFyZ2V0` |
| `training.field.attestation_revision` | `cert_version` | `Y2VydF92ZXJz…` |
| `training.field.attestation_version` | `certificate_version` | `Y2VydGlmaWNh…` |
| `training.field.batch_identifier` | `job_id` | `am9iX2lk` |
| `training.field.batch_seed` | `nonce` | `bm9uY2U=` |
| `training.field.compact_acceptance_bound` | `nbits` | `bmJpdHM=` |
| `training.field.coordinator_work_identifier` | `workid` | `d29ya2lk` |
| `training.field.dense_attestation` | `plain_proof` | `cGxhaW5fcHJv…` |
| `training.field.input_row_index_label` | `a_rows index` | `YV9yb3dzIGlu…` |
| `training.field.input_row_indices` | `a_rows` | `YV9yb3dz` |
| `training.field.matrix_tile_height` | `tile_h` | `dGlsZV9o` |
| `training.field.matrix_tile_width` | `tile_w` | `dGlsZV93` |
| `training.field.parent_record_digest` | `prev_block` | `cHJldl9ibG9j…` |
| `training.field.signal_baseline` | `sb` | `c2I=` |
| `training.field.signal_baseline_parameters` | `sb_p` | `c2JfcA==` |
| `training.field.tensor_commitment_root` | `merkle_root` | `bWVya2xlX3Jv…` |
| `training.field.tensor_tail_mode` | `tail_mode` | `dGFpbF9tb2Rl` |
| `training.field.training_worker` | `worker` | `d29ya2Vy` |
| `training.field.weight_column_index_label` | `b_cols index` | `Yl9jb2xzIGlu…` |
| `training.field.weight_column_indices` | `b_cols` | `Yl9jb2xz` |
| `training.fixture.byte_sweep` | `000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff` (256 bytes, hex) | `AAECAwQFBgcI…` |
| `training.fixture.empty_input` | `""` | `""` |
| `training.fixture.text_sample` | `abc` | `YWJj` |
| `training.format.attestation_payload_tag` | `aa21a9ed` | `YWEyMWE5ZWQ=` |
| `training.format.batch_header_mask` | `9d2f73c411a85be0367ad14e920cbf65f32884da176eb941cc530fa67de21974` (32 bytes, hex) | `nS9zxBGoW+A2…` |
| `training.format.batch_update_prefix_mask` | `4dc0d47dff723cf8f75a583b596cf848f5106dd0d0aa7bdded9d4d566a929ffaabea8985cf34a26af1b118e2c697f283f0caadd643b0975390a17c79aae32e69cf166b57a6f64dd02914e7e4` (76 bytes, hex) | `TcDUff9yPPj3…` |
| `training.format.empty_parent_output_index` | `ffffffff` | `ZmZmZmZmZmY=` |
| `training.label.attestation_integer` | `proof integer` | `cHJvb2YgaW50…` |
| `training.label.attestation_version` | `certificate version` | `Y2VydGlmaWNh…` |
| `training.label.attestation_version_config_path` | `{}.certificate_version` | `e30uY2VydGlm…` |
| `training.label.batch_acceptance_bound` | `share target` | `c2hhcmUgdGFy…` |
| `training.label.batch_update_payload` | `notify payload` | `bm90aWZ5IHBh…` |
| `training.label.legacy_batch_protocol_prefix` | `"S7 "` | `Uzcg` |
| `training.label.tensor_commitment_tree` | `Merkle` | `TWVya2xl` |
| `training.log.accelerator_duty_cycle` | `%s effort GPU duty-cycle target: %d%%` | `JXMgZWZmb3J0…` |
| `training.log.accelerator_metrics_acknowledged` | `S7 acknowledged GPU information report` | `UzcgYWNrbm93…` |
| `training.log.accelerator_metrics_acknowledgement_missing` | `S7 discarded an unacknowledged GPU report` | `UzcgZGlzY2Fy…` |
| `training.log.accelerator_metrics_rejected` | `S7 rejected GPU information report` | `UzcgcmVqZWN0…` |
| `training.log.accelerator_metrics_sent` | `S7 GPU report sent: %s (request=%d)` | `UzcgR1BVIHJl…` |
| `training.log.accepted_batch_tile_location` | `"Found valid block at "` | `Rm91bmQgdmFs…` |
| `training.log.attestation_record_assembly` | `Generating block from PlainProof` | `R2VuZXJhdGlu…` |
| `training.log.attestation_record_generated` | `Generated block` | `R2VuZXJhdGVk…` |
| `training.log.batch_generation` | `S7 job=%s generation=%d` | `Uzcgam9iPSVz…` |
| `training.log.batch_header_verified_no_compute` | `Authenticated job verified: header=%d bytes, target=%d bits; no GPU work performed` | `QXV0aGVudGlj…` |
| `training.log.batch_header_verified_no_submissions` | `Authenticated job verified: header=%d bytes, target=%d bits; no GPU work or shares submitted` | `QXV0aGVudGlj…` |
| `training.log.batch_solution_verified` | `S1 solution verified successfully` | `UzEgc29sdXRp…` |
| `training.log.batch_submission_accepted` | `S3 accepted training share job=%s` | `UzMgYWNjZXB0…` |
| `training.log.batch_submission_rejected` | `S3 rejected training share job=%s: %s` | `UzMgcmVqZWN0…` |
| `training.log.coordinator_batch_revision` | `S3 job=%s height=%d certificate=V%d` | `UzMgam9iPSVz…` |
| `training.log.coordinator_session_authenticated` | `S3 authenticated at %s:%d; proof compression=%s` | `UzMgYXV0aGVu…` |
| `training.log.training_run_summary` | `S3 training stopped: batches=%d submitted=%d accepted=%d rejected=%d stale=%d` | `UzMgdHJhaW5p…` |
| `training.log.zero_knowledge_attestation_generated` | `Generated ZK proof` | `R2VuZXJhdGVk…` |
| `training.log.zero_knowledge_attestation_verified` | `verified ZK proof` | `dmVyaWZpZWQg…` |
| `training.log.zero_knowledge_attestation_verifying` | `verifying ZK proof` | `dmVyaWZ5aW5n…` |
| `training.message.accelerator_metrics` | `report_info` | `cmVwb3J0X2lu…` |
| `training.message.batch_update` | `notify` | `bm90aWZ5` |
| `training.message.coordinator_reconnect` | `client.reconnect` | `Y2xpZW50LnJl…` |
| `training.message.open_session` | `login` | `bG9naW4=` |
| `training.message.submit_batch_result` | `submit` | `c3VibWl0` |
| `training.metadata.client_agent` | `LRAEP-trainer/0.1` | `TFJBRVAtdHJh…` |
| `training.metadata.trainer_title` | `LRAEP-WR-Trainer` | `TFJBRVAtV1It…` |
| `training.path.unrecognized_batch_messages` | `unknown-S3-requests` | `dW5rbm93bi1T…` |
| `training.tensor.input_noise_domain` | `415f74656e736f72000000000000000000000000000000000000000000000000` (32 bytes, hex) | `QV90ZW5zb3IA…` |
| `training.tensor.left_operand_label` | `A` | `QQ==` |
| `training.tensor.right_operand_label` | `B` | `Qg==` |
| `training.tensor.weight_noise_domain` | `425f74656e736f72000000000000000000000000000000000000000000000000` (32 bytes, hex) | `Ql90ZW5zb3IA…` |
| `training.thread.batch_message_reader` | `WR-S3-reader` | `V1ItUzMtcmVh…` |
| `training.thread.batch_session_stop_monitor` | `WR-S3-stop` | `V1ItUzMtc3Rv…` |
| `training.validation.accelerator_attestation_digest_length` | `GPU share digest must be exactly 32 bytes` | `R1BVIHNoYXJl…` |
| `training.validation.accelerator_batch_digest_mismatch` | `GPU S7 digest differs from independently reconstructed CPU work` | `R1BVIFM3IGRp…` |
| `training.validation.accelerator_metrics_acknowledgement_timeout` | `S7 GPU report acknowledgement timed out` | `UzcgR1BVIHJl…` |
| `training.validation.accelerator_metrics_authentication_required` | `S7 session must authenticate before reporting GPU information` | `Uzcgc2Vzc2lv…` |
| `training.validation.accelerator_metrics_send_failed` | `S7 GPU report send failed` | `UzcgR1BVIHJl…` |
| `training.validation.accelerator_reference_digest_mismatch` | `GPU share digest differs from the official CPU reference` | `R1BVIHNoYXJl…` |
| `training.validation.accelerator_required` | `an available CUDA GPU is required for S3 training` | `YW4gYXZhaWxh…` |
| `training.validation.accelerator_tensor_commitment_mismatch` | `native GPU Merkle root differs from the official CPU tree` | `bmF0aXZlIEdQ…` |
| `training.validation.accelerator_throughput_range` | `hashrate must be a nonnegative finite number in H/s` | `aGFzaHJhdGUg…` |
| `training.validation.acceptance_bound_bit_length` | `target must fit in 256 bits` | `dGFyZ2V0IG11…` |
| `training.validation.acceptance_bound_hex_format` | `target must be a hexadecimal string` | `dGFyZ2V0IG11…` |
| `training.validation.acceptance_bound_interval_prefix` | `pow_target must be in the range pow_target=` | `cG93X3Rhcmdl…` |
| `training.validation.acceptance_bound_positive_hex_range` | `target must be a positive 256-bit hexadecimal string` | `dGFyZ2V0IG11…` |
| `training.validation.acceptance_bound_scaling_overflow` | `target scaling overflows; choose a harder test-network target` | `dGFyZ2V0IHNj…` |
| `training.validation.account_identity_required_suffix` | `" must contain a valid S6 Taproot address"` | `IG11c3QgY29u…` |
| `training.validation.attestation_commitment_length_prefix` | `"Proof commitment must be "` | `UHJvb2YgY29t…` |
| `training.validation.attestation_commitment_mismatch` | `Proof commitment mismatch` | `UHJvb2YgY29t…` |
| `training.validation.attestation_commitment_missing` | `Proof commitment is not set` | `UHJvb2YgY29t…` |
| `training.validation.attestation_digest_outside_bound` | `proof digest does not meet difficulty target` | `cHJvb2YgZGln…` |
| `training.validation.attestation_input_size_limit` | `training proof inputs exceed S6's 4 MiB worker-input limit` | `dHJhaW5pbmcg…` |
| `training.validation.attestation_matrix_rows_invalid` | `invalid proof matrix rows` | `aW52YWxpZCBw…` |
| `training.validation.attestation_pattern_offset_invalid` | `invalid proof pattern offset` | `aW52YWxpZCBw…` |
| `training.validation.attestation_rank_or_inner_dimension` | `unsupported proof rank or common dimension` | `dW5zdXBwb3J0…` |
| `training.validation.attestation_revision_too_old` | `proof requires a later certificate version` | `cHJvb2YgcmVx…` |
| `training.validation.attestation_revision_unsupported` | `unsupported certificate version` | `dW5zdXBwb3J0…` |
| `training.validation.attestation_selected_tile_size_limit` | `selected proof rows and columns exceed S6's 4 MiB worker-input limit` | `c2VsZWN0ZWQg…` |
| `training.validation.attestation_tensor_signal_range` | `proof signal values must lie in [-64, 64]` | `cHJvb2Ygc2ln…` |
| `training.validation.attestation_tile_shape_invalid` | `invalid proof tile shape` | `aW52YWxpZCBw…` |
| `training.validation.attestation_truncated` | `truncated proof` | `dHJ1bmNhdGVk…` |
| `training.validation.attestation_vector_length_invalid` | `invalid proof vector length` | `aW52YWxpZCBw…` |
| `training.validation.attestation_version_prefix` | `"unsupported S6 certificate version "` | `dW5zdXBwb3J0…` |
| `training.validation.attestation_version_template` | `unsupported S6 certificate version {}` | `dW5zdXBwb3J0…` |
| `training.validation.authenticated_batch_required` | `no authenticated S3 job was received before stopping` | `bm8gYXV0aGVu…` |
| `training.validation.batch_acceptance_bound_range` | `S7 final target must be a positive uint256` | `UzcgZmluYWwg…` |
| `training.validation.batch_account_identity_whitespace` | `WR_S2 and worker must contain no whitespace` | `V1JfUzIgYW5k…` |
| `training.validation.batch_and_compact_attestation_required` | `S7 submission requires its job and compact share` | `Uzcgc3VibWlz…` |
| `training.validation.batch_attestation_revision_required` | `job is missing cert_version; configure it explicitly or update the S3` | `am9iIGlzIG1p…` |
| `training.validation.batch_candidate_digest_length` | `S7 candidate digest must contain exactly 32 bytes` | `UzcgY2FuZGlk…` |
| `training.validation.batch_candidate_outside_acceptance_bound` | `S7 candidate does not meet the exact S3 target` | `UzcgY2FuZGlk…` |
| `training.validation.batch_compression_unsupported` | `S7 compression must be none` | `UzcgY29tcHJl…` |
| `training.validation.batch_device_seed_length` | `S7 worker nonce must contain eight bytes` | `Uzcgd29ya2Vy…` |
| `training.validation.batch_dimensions_or_signal_mode` | `Unsupported S7 work dimensions or signal mode` | `VW5zdXBwb3J0…` |
| `training.validation.batch_generation_attestation_revision` | `S7 generation 3 requires certificate version 3` | `UzcgZ2VuZXJh…` |
| `training.validation.batch_generation_three_required` | `S7 training requires an observed generation-3 job` | `UzcgdHJhaW5p…` |
| `training.validation.batch_generation_unsupported` | `Only observed S7 generation 3 is supported` | `T25seSBvYnNl…` |
| `training.validation.batch_header_compact_bound_invalid` | `S7 decoded header has an invalid compact target` | `UzcgZGVjb2Rl…` |
| `training.validation.batch_header_length` | `S7 header must contain exactly 76 bytes` | `UzcgaGVhZGVy…` |
| `training.validation.batch_notification_name_malformed` | `Malformed S7 notification name` | `TWFsZm9ybWVk…` |
| `training.validation.batch_replaced_before_submission` | `S3 replaced this training job before submission` | `UzMgcmVwbGFj…` |
| `training.validation.batch_request_identifier_range` | `S7 request ID must be an unsigned 64-bit integer` | `UzcgcmVxdWVz…` |
| `training.validation.batch_request_unknown` | `Unknown S7 request; training stopped` | `VW5rbm93biBT…` |
| `training.validation.batch_response_envelope_unknown` | `Unknown S7 response envelope` | `VW5rbm93biBT…` |
| `training.validation.batch_seed_pair_length` | `S7 nonces must each contain exactly eight bytes` | `Uzcgbm9uY2Vz…` |
| `training.validation.batch_server_fields_unknown` | `Unknown S7 server message fields or method` | `VW5rbm93biBT…` |
| `training.validation.batch_server_message_shape_unknown` | `Unknown S7 server request or response shape` | `VW5rbm93biBT…` |
| `training.validation.batch_server_method_unknown` | `Unknown S7 server method` | `VW5rbm93biBT…` |
| `training.validation.batch_server_notification_unknown` | `Unknown S7 server notification` | `VW5rbm93biBT…` |
| `training.validation.batch_session_authentication_rejected` | `S7 rejected authentication` | `UzcgcmVqZWN0…` |
| `training.validation.batch_submission_compression_unsupported` | `S7 does not support KEX gzip proof submissions` | `UzcgZG9lcyBu…` |
| `training.validation.batch_submission_device_seed_mismatch` | `S7 share worker nonce does not match its job` | `Uzcgc2hhcmUg…` |
| `training.validation.batch_submission_dimensions_type` | `S7 share dimensions must be integers` | `Uzcgc2hhcmUg…` |
| `training.validation.batch_submission_fields_invalid` | `Invalid S7 share fields` | `SW52YWxpZCBT…` |
| `training.validation.batch_submission_outside_acceptance_bound` | `training share does not meet the exact S3 target` | `dHJhaW5pbmcg…` |
| `training.validation.batch_tile_coordinate_range` | `S7 tile coordinates are out of range` | `UzcgdGlsZSBj…` |
| `training.validation.batch_tile_index_order_and_seed_row` | `S7 tile indices must be sorted, unique and outside the nonce row` | `UzcgdGlsZSBp…` |
| `training.validation.batch_update_payload_length` | `S7 notify payload must contain 200 bytes` | `Uzcgbm90aWZ5…` |
| `training.validation.batch_update_schema` | `Unsupported S7 notify schema` | `VW5zdXBwb3J0…` |
| `training.validation.compact_acceptance_bound_invalid` | `Invalid compact target` | `SW52YWxpZCBj…` |
| `training.validation.compact_attestation_required` | `S7 verification requires a compact share` | `UzcgdmVyaWZp…` |
| `training.validation.coordinator_acceptance_bound_kind` | `S3 target kind must be base or final` | `UzMgdGFyZ2V0…` |
| `training.validation.coordinator_acceptance_bound_range` | `node target must be a positive 256-bit integer` | `bm9kZSB0YXJn…` |
| `training.validation.coordinator_acceptance_bound_value` | `S3 target must be a positive 256-bit integer` | `UzMgdGFyZ2V0…` |
| `training.validation.coordinator_authentication_rejected` | `S3 rejected authentication` | `UzMgcmVqZWN0…` |
| `training.validation.coordinator_authentication_timeout` | `S3 authentication timed out` | `UzMgYXV0aGVu…` |
| `training.validation.coordinator_batch_timeout` | `S3 sent no usable training job before the job timeout` | `UzMgc2VudCBu…` |
| `training.validation.coordinator_closed_before_acknowledgement` | `S3 closed before acknowledging the share` | `UzMgY2xvc2Vk…` |
| `training.validation.coordinator_compression_unsupported` | `S3 compression must be none or gzip` | `UzMgY29tcHJl…` |
| `training.validation.coordinator_config_prefix` | `fill S3.` | `ZmlsbCBTMy4=` |
| `training.validation.coordinator_host_config_prefix` | `fill S3.host for a flat configuration, or S3.` | `ZmlsbCBTMy5o…` |
| `training.validation.coordinator_hostname_suffix` | `.host in the shared configuration, with your S3 hostname or IP address` | `Lmhvc3QgaW4g…` |
| `training.validation.coordinator_identity_whitespace_template` | `S3.{} and S3.worker must not contain whitespace` | `UzMue30gYW5k…` |
| `training.validation.coordinator_pending_submission_limit` | `Too many pending S3 submissions` | `VG9vIG1hbnkg…` |
| `training.validation.coordinator_protocol_explicit_selection` | `S3.protocol must explicitly select KEX or WR; automatic detection is unavailable` | `UzMucHJvdG9j…` |
| `training.validation.coordinator_protocol_unsupported` | `S3.protocol must select KEX or WR` | `UzMucHJvdG9j…` |
| `training.validation.coordinator_reconnect_requested` | `S3 requested reconnection to the configured endpoint` | `UzMgcmVxdWVz…` |
| `training.validation.coordinator_seed_required` | `S7 training requires the S3's eight-byte extranonce` | `UzcgdHJhaW5p…` |
| `training.validation.coordinator_seed_update_schema` | `Unsupported S7 extranonce schema` | `VW5zdXBwb3J0…` |
| `training.validation.coordinator_session_already_stopped` | `S3 session is stopped` | `UzMgc2Vzc2lv…` |
| `training.validation.coordinator_session_closed` | `S3 session closed` | `UzMgc2Vzc2lv…` |
| `training.validation.coordinator_session_not_open` | `S3 session is not open` | `UzMgc2Vzc2lv…` |
| `training.validation.coordinator_session_opening` | `S3 session is already opening` | `UzMgc2Vzc2lv…` |
| `training.validation.coordinator_session_stopped` | `S3 session stopped` | `UzMgc2Vzc2lv…` |
| `training.validation.coordinator_submission_acknowledgement_timeout` | `S3 share acknowledgement timed out` | `UzMgc2hhcmUg…` |
| `training.validation.coordinator_submission_attestation_version` | `unsupported S6 certificate version for S3 share` | `dW5zdXBwb3J0…` |
| `training.validation.coordinator_submission_rejected` | `S3 rejected share` | `UzMgcmVqZWN0…` |
| `training.validation.coordinator_submission_send_failed` | `S3 share send failed` | `UzMgc2hhcmUg…` |
| `training.validation.coordinator_submission_verification_prefix` | `"S3 share rejected by official verification: "` | `UzMgc2hhcmUg…` |
| `training.validation.coordinator_tile_coordinates_type` | `S3 share tile coordinates must be integers` | `UzMgc2hhcmUg…` |
| `training.validation.coordinator_tls_authority_unreadable` | `Could not load RPC certificate authority file` | `Q291bGQgbm90…` |
| `training.validation.coordinator_tls_url_required` | `A certificate authority file requires an https RPC URL` | `QSBjZXJ0aWZp…` |
| `training.validation.coordinator_wait_timeout_finite` | `S3 wait timeout must be a finite number` | `UzMgd2FpdCB0…` |
| `training.validation.coordinator_wait_timeout_positive` | `S3 wait timeout must be a positive finite number` | `UzMgd2FpdCB0…` |
| `training.validation.coordinator_worker_identity_whitespace` | `KEX_uname and worker must contain no whitespace` | `S0VYX3VuYW1l…` |
| `training.validation.cpu_attestation_tile_incomplete` | `official CPU reference produced no complete share tile` | `b2ZmaWNpYWwg…` |
| `training.validation.cpu_batch_tile_incomplete` | `CPU reference produced no complete S7 tile` | `Q1BVIHJlZmVy…` |
| `training.validation.dense_attestation_canonical_format` | `plain_proof must be nonempty canonical base64` | `cGxhaW5fcHJv…` |
| `training.validation.dense_attestation_encoding_invalid` | `invalid plain proof encoding` | `aW52YWxpZCBw…` |
| `training.validation.dense_attestation_input_format` | `plain proof must be a bounded base64 string` | `cGxhaW4gcHJv…` |
| `training.validation.dense_attestation_size_limit` | `plain_proof exceeds the 4 MiB decoded size limit` | `cGxhaW5fcHJv…` |
| `training.validation.dense_attestation_trailing_data` | `unsupported or trailing plain proof data` | `dW5zdXBwb3J0…` |
| `training.validation.dense_model_attestation_required` | `MoE proofs are rejected by the dense-only rule` | `TW9FIHByb29m…` |
| `training.validation.final_acceptance_bound_range` | `final S3 target must be a positive 256-bit integer` | `ZmluYWwgUzMg…` |
| `training.validation.local_attestation_generation_unavailable` | `Local ZK certificate generation is unavailable; WR and plain S3 shares do not require it` | `TG9jYWwgWksg…` |
| `training.validation.local_attestation_verification_unavailable` | `Local ZK certificate verification is unavailable` | `TG9jYWwgWksg…` |
| `training.validation.mixture_of_experts_attestation_unavailable` | `MoE plain proofs are unavailable` | `TW9FIHBsYWlu…` |
| `training.validation.tensor_commitment_header_mismatch` | `Merkle proof commitment does not match header and configuration` | `TWVya2xlIHBy…` |
| `training.validation.tensor_commitment_input_empty` | `Merkle data must not be empty` | `TWVya2xlIGRh…` |
| `training.validation.tensor_commitment_leaf_count_mismatch` | `Merkle tree leaf count differs from declared dimensions` | `TWVya2xlIHRy…` |
| `training.validation.tensor_commitment_leaf_count_positive` | `Merkle tree requires a positive leaf count` | `TWVya2xlIHRy…` |
| `training.validation.tensor_commitment_leaf_order` | `Merkle leaf indices must be sorted, unique and in range` | `TWVya2xlIGxl…` |
| `training.validation.tensor_commitment_leaf_range` | `Merkle leaf indices are out of range` | `TWVya2xlIGxl…` |
| `training.validation.tensor_commitment_matrix_coverage` | `Merkle leaves do not cover required matrix bytes` | `TWVya2xlIGxl…` |
| `training.validation.tensor_commitment_slice_range` | `invalid Merkle extraction range` | `aW52YWxpZCBN…` |
| `training.validation.transcript_digest_parameters_invalid` | `invalid transcript hash key or target` | `aW52YWxpZCB0…` |
| `training.validation.worker_identity_type` | `S3.worker must be a string` | `UzMud29ya2Vy…` |
| `training.validation.worker_identity_whitespace_suffix` | `" and S3.worker must not contain whitespace"` | `IGFuZCBTMy53…` |

Text values with leading or trailing spaces are quoted to show the exact
spacing. Raw binary values are shown in hex. Base64 previews show at most
12 characters followed by `…` when shortened; they preview the Base64
stored in the native table. Shortened previews cannot be used as decoder inputs. RPC fields and methods use the
structural entries selected by the native wire profile. Explicit constant
aliases are resolved by their callers and do not rewrite arbitrary message
content. The aliases describe internal names and preserve the original peer
protocol and proof behavior.

### Updating and distributing mappings

1. In ANN-Trainer, run `python3 tools/build_word_codec.py --encode-value ALIAS VALUE`
   to print an alias/Base64 row; add `--binary` when VALUE is hexadecimal
   binary data. Add the resulting `make_base64_entry` or
   `make_base64_binary_entry` row to `native/word_codec.cpp`.
   Aliases and original values must each remain unique. For an explicit constant,
   add the row to `kConstants` and use `FixedAlias` in Python. For a new
   network field or command, also select it in the appropriate `key_entry` or
   `Parser::translation` profile in `native/wire_codec.cpp`. Profiles distinguish
   RPC fields, pool fields, command names and ordinary string data; adding a
   word-table entry alone does not rewrite arbitrary message values.
2. Run `python3 tools/build_word_codec.py`. It validates the table, compiles the
   word and wire libraries, regenerates this table in all three READMEs, and atomically
   installs identical libraries and Python bridges. This includes
   `native_text_codec.py`, `wire_io.py` and `rpc_fields.py`. The default standalone
   checkout is the sibling `WR-trainer` (or `LRAEP-WR-Trainer` when present);
   the production checkout is the sibling `Production-LREAP` when present.
   Use `--standalone PATH` and `--production PATH` to choose other checkouts,
   or `--local-only` for an author-only build. Only compiled artifacts and
   Python bridges are copied; native C++ source stays in the trainer checkouts.
3. Run `python3 tools/build_word_codec.py --check` and the protocol/native tests.
   The check detects stale README content, source fingerprints, differing
   binaries, or differing Python bridges. Restart running trainer/notebook
   processes to load a newly installed library; loaded libraries are not
   hot-swapped in place.

The current build targets the host's Linux architecture (`x86_64` or
`aarch64`) and requires a C++17 compiler. The output is
`src/LRAEP_trainer/bin/<platform>/libword_codec.so` and, when distributing,
`src/workrunner/bin/<platform>/libword_codec.so`. The native message layer is
installed alongside it as `libwire_codec.so`. Rebuild on each target
architecture. No compiler or C++ decoder source is needed in the standalone
at runtime. A missing or incompatible library produces an explicit error.

### Buffer lifetime

`word_buffer` and `fixed_buffer` copy values into mutable buffers and clear
the buffers on normal and exception exits. `word_matches` compares ASCII
protocol values with a scoped buffer. Native Base64 and decoded comparison
buffers are also cleared immediately on success and exception exits with volatile
writes, including wire/frame temporaries. The native table retains
only the Base64 bytes; plaintext is recovered separately for each call.

`transform_word`, `decode_word`, and `encode_word` return ordinary immutable
Python strings. Callers must release those references after use; their physical
bytes cannot be guaranteed erased, and serializers, sockets, exception text,
or caller-retained objects may create copies. The framing-only
`customEncoder`/`customDecoder` API implements the peer's frame format.
<!-- native-word-codec:end -->
