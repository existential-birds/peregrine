# peregrine

Local LLM inference-optimisation lab on Apple Silicon. We hill-climb serving
speed — decode tok/s, prefill, TTFT, startup, KV — the way the peregrine
hunts: compute the ceiling first, then dive as close to it as the physics
allows.

Built around the [inference-optimisation
skill](skills/inference-optimisation/SKILL.md) (vendored from
[sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill),
MIT, with our MLX/Apple-Silicon additions in
[references/mlx.md](skills/inference-optimisation/references/mlx.md)):

1. Measure the hardware; compute the theoretical ceiling before optimising.
2. Change one thing per experiment.
3. Benchmark any OpenAI-compatible server with interleaved controls and
   medians (`scripts/bench.py`, `ledger.jsonl`).
4. Gate every "best" on long-context needle retrieval and a full-KV soak.

## Current target

**Inkling-2-Mini** (Thinking Machines' looped-layer MoE VLM) served by MLX
via our fork [existential-birds/mlx-vlm](https://github.com/existential-birds/mlx-vlm)
(upstream: Neywa-Labs/mlx-vlm). ~145 tok/s c1 ceiling on this machine —
novelty over speed was a deliberate trade. The plan, open questions, and
pinned references are in [PROPOSAL.md](PROPOSAL.md); the experiment log is
[PLAN.md](PLAN.md); measured hardware and ceilings are in
[THEORY.md](THEORY.md).

## Layout

| Path | What |
|---|---|
| `PROPOSAL.md` | The current project: scope, verified facts, research questions, phases |
| `PLAN.md` | Timestamped experiment log (targets, inventory, results, failures) |
| `THEORY.md` | Measured hardware and theoretical ceilings |
| `configs/` | Bench configs and ceiling inputs, one JSON per run |
| `scripts/` | Local helpers: server launch/stop, macOS memory probe, Metal measurement |
| `skills/inference-optimisation/` | The vendored skill (procedure, references, harness) |
| `reference/llama-cpp/` | Vendored llama.cpp `inkling2-loop` reference implementation |

## Hardware (measured, not spec sheets)

MacBook Pro, Apple M4 Max (40-core GPU), 48 GB unified memory:
**440 GB/s** device copy, **15.2 TFLOPS** fp16 GEMM
(`scripts/measure_metal.py`, 2026-11-25). Benchmarks run on AC power only,
under a 36 GiB memory cap with a footprint watchdog.
