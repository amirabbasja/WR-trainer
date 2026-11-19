# WR Trainer

The WR client follows the PearlHash exchange captured from WildRig Multi 0.52.3.
It starts each connection with login request ID `0` and defaults to the agent
`WildRig_0.52.3`. An explicitly configured custom agent still takes precedence.

The first `report_info` is due 15 seconds after login. Subsequent reports are due
120.6 seconds apart. The trainer sends a due report after completing a batch.

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
