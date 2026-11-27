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
via our fork [existential-birds/mlx-vlm](https://github.com/existential-birds/mlx-vlm/tree/perf/inkling2-serving)
(`perf/inkling2-serving`, upstream: `Neywa-Labs/mlx-vlm`).

- **Published checkpoints (Hugging Face, Apache 2.0):**
  - Target VLM (`17.27 GB`, mixed affine 4-bit dense + native NVFP4/MXFP8 experts): [`existentialbirds/Inkling-2-Mini-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-4bit)
  - Companion 4-bit DSpark speculative drafter (`247 MiB`): [`existentialbirds/Inkling-2-Mini-MTP-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-MTP-4bit)
- **Qualified results (`best-config.json` on M4 Max 48 GB):** **`46.5 tok/s`** `c1` decode, **`93.3 tok/s`** `c4` aggregate decode, **`590.5 / 547.4 tok/s`** cold prefill at 4K/32K, **3/3 exact needles at 32K and 131K**, and **6/6 concurrent 134.7K KV soak** inside `23.9–30.1 GiB`.
- Full research plan in [PROPOSAL.md](PROPOSAL.md), experiment log & 10-stage Metal profiling in [PLAN.md](PLAN.md), measured hardware & sparse-MoE ceilings in [THEORY.md](THEORY.md), and raw interleaved runs in [ledger.jsonl](ledger.jsonl).

## Layout

| Path | What |
|---|---|
| `PROPOSAL.md` | The current project: scope, verified facts, research questions, phases |
| `PLAN.md` | Timestamped experiment log (targets, inventory, results, failures) |
| `THEORY.md` | Measured hardware and theoretical ceilings |
| `best-config.json` | Qualified winning launch configuration |
| `ledger.jsonl` | Append-only JSONL ledger of interleaved benchmark and gate runs |
| `configs/` | Bench configs and ceiling inputs, one JSON per run |
| `scripts/` | Local helpers: server launch/stop, macOS memory probe, Metal measurement |
| `skills/inference-optimisation/` | The vendored skill (procedure, references, harness) |
| `reference/llama-cpp/` | Vendored llama.cpp `inkling2-loop` reference implementation |
| `THIRD_PARTY_LICENSES.md` | MIT copyright and license notices for vendored files |

## Hardware (measured, not spec sheets)

MacBook Pro, Apple M4 Max (40-core GPU), 48 GB unified memory:
**440 GB/s** device copy, **15.2 TFLOPS** fp16 GEMM
(`scripts/measure_metal.py`, 2026-11-25). Benchmarks run on AC power only,
under a 36 GiB memory cap with a footprint watchdog.
