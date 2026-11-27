# Peregrine

Local LLM inference-optimisation lab on Apple Silicon. We hill-climb serving
speed — decode `tok/s`, prefill, TTFT, startup, and KV footprint — the way the
peregrine hunts: compute the ceiling from measured hardware first, then dive as
close to it as the physics allows.

Built on the [inference-optimisation skill](skills/inference-optimisation/SKILL.md)
(adapted from [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill)
with Apple Silicon / MLX additions in [`references/mlx.md`](skills/inference-optimisation/references/mlx.md)):

1. **Measure the hardware** (`scripts/measure_metal.py`) and derive byte/FLOP ceilings before touching flags or kernels.
2. **Change one variable per experiment** and keep failed hypotheses in the log.
3. **Interleave controls** around every candidate (`scripts/queue.sh`, medians of ≥6 in [`ledger.jsonl`](ledger.jsonl)).
4. **Gate every winner** on logit fidelity, 32K & 131K exact-match needle retrieval, concurrent full-KV soak, and multimodal/tool round-trips under a hard memory cap.

---

## Case Study: `Inkling-2-Mini` on Apple M4 Max (48 GB)

Fast, validated local serving of Thinking Machines' [`Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) — a 26.4B-parameter looped-layer sparse Mixture-of-Experts VLM (`3.4B` active per pass; middle 12 layers run 3 passes for **48 layer visits** and `~5.4B` effective active parameters per token) — using [`existential-birds/mlx-vlm`](https://github.com/existential-birds/mlx-vlm/tree/perf/inkling2-serving) (`perf/inkling2-serving`).

### Published Checkpoints & Runtime

| Artifact | Link | Description |
|---|---|---|
| **4-bit VLM (`17.27 GB`)** | [`existentialbirds/Inkling-2-Mini-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-4bit) | `mlx-vlm`-native checkpoint: affine 4-bit (`g=64`) dense text layers + untouched native `nvfp4` (L0–19) & `mxfp8` (L20–23) routed experts + BF16 vision |
| **4-bit DSpark Drafter (`247 MiB`)** | [`existentialbirds/Inkling-2-Mini-MTP-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-MTP-4bit) | Standalone 3-layer DSpark block-diffusion speculative drafter extracted from `mtp.safetensors` |
| **MLX-VLM Fork** | [`existential-birds/mlx-vlm` (`perf/inkling2-serving`)](https://github.com/existential-birds/mlx-vlm/tree/perf/inkling2-serving) | Lossless batch-cache, loop-controller, attention-sink, and `--cache-limit` fastpaths (`f871d0b7`) |

### Measured Results (Apple M4 Max, 40-Core GPU, 48 GB, macOS 26.6.1)

Measured on AC power (`mlx 0.32.5`, `T=0.6`, `reasoning_effort="none"`, 6-prompt fixed panel, server-reported token counts):

| Metric | Official NVFP4 (`thinkingmachines/Inkling-2-Mini`) | Stock `mlx-vlm` + 4-bit Checkpoint | Tuned ([`best-config.json`](best-config.json)) | Practical Hardware Ceiling |
|---|---:|---:|---:|---:|
| **`c1` decode (median of 6)** | `33.1 tok/s` | `44.1 tok/s` | **`46.5 tok/s`** | `139 tok/s` (raw BW @ 4K) |
| **`c4` aggregate decode** | `16.2–34.7 tok/s` *(swap)* | `79.4 tok/s` | **`93.3 tok/s`** (`4 × 24.0`) | `224 tok/s` (sparse-MoE @ 4K) |
| **Cold prefill @ 3.3K tok** | `332.8 tok/s` | `551.1 tok/s` | **`590.5 tok/s`** | `727 tok/s` (50% MFU) |
| **Cold prefill @ 25.6K tok** | `306.7–376.0 tok/s` | `524.1 tok/s` | **`547.4 tok/s`** | `641 tok/s` (50% MFU) |
| **Warm short TTFT / Startup** | `245 ms` / `9.2 s` | `198 ms` / `7.3 s` | **`196 ms` / `5.3 s`** | — |
| **Peak server memory (≤33K ctx)** | `34.30 GiB` | `29.93 GiB` | **`23.91 GiB`** | `36.0 GiB` cap |

**Qualification gates on [`best-config.json`](best-config.json) (all passed under the 36 GiB cap, `iogpu.wired_limit_mb=0` untouched):**
- **Fidelity vs native NVFP4 (`T=0`):** first-token logit cosine `0.9986–0.9996`, softmax `KL = 0.00000 nats`, `3/3` top-1 match, `51–73` exact greedy prefix tokens before first BF16 tie.
- **32K needle retrieval (`33,248` tokens):** **`3/3` exact-match** codes (`458.1 tok/s` prefill, `26.54 tok/s` decode, `23.91 GiB`).
- **131K needle retrieval (`132,777` tokens):** **`3/3` exact-match** codes (`258.9 tok/s` prefill, `22.74 tok/s` decode, `30.08 GiB`).
- **Concurrent full-KV soak (`2 × 67.3K = 134,684` live tokens):** **`6/6` exact-match** codes across 2 simultaneous streams (`25.18 GiB`, zero looping).
- **Modalities:** multi-step reasoning, structured `inkling` tool-call round-trips, and vision (`image_url`) verified.

### Key Findings (Full Details in [`PLAN.md`](PLAN.md) & [`THEORY.md`](THEORY.md))

1. **Where the `139 → 46.5 tok/s` `c1` gap actually goes:** Quantizing dense attention + `lm_head` to 4-bit cuts per-token weight reads by **2.08×** (`6.30 → 3.03 GB/tok`) while keeping native `nvfp4`/`mxfp8` MoE kernels untouched. In a 10-stage GPU profile (`23.62 ms/tok`), weight kernels take **`9.92 ms`** (`70%` of the `6.89 ms` DRAM floor), while **`13.70 ms` (`58%`)** goes to non-weight loop & Metal serial dispatch across **`9,951` MLX ops** (`MessageController`s, 97 `AttentionResidual` `mx.stack` + softmax reductions, and 192 short-convolutions).
2. **Sparse-MoE concurrency physics (`224 tok/s`, not `836 tok/s`):** Standard batch-ceiling formulas assume weights are read once per batch. With top-8 of 128 routing, 4 independent streams touch `128 × (1 - (120/128)^4) = 29.0` unique experts per layer (`3.62×` single-stream expert bytes = `7.85 GB/step`).
3. **Four lossless `mlx-vlm` fastpaths (`+15.8%` rested / `+36.9%` warm `c4`):** Auto-clearing spent `ArraysCache.left_padding` after prefill removed **`+7,368` fallback ops/step** (`+15.1 ms/step`) across 176 short-convolutions on unequal-length batches; adding zero-copy `BatchKVCache.with_sink`, batched `@mx.compile`d `MessageController`s, and `batch * rows <= 8` head-prep Metal dispatch lifted `c4` to `93.3 tok/s`.
4. **131K prefill mask scaling & DSpark speculative break-even:** Inkling-2's banded relative-position masks allocate `[B, 24, prefill_step_size, context_len]` on global visits (`24.3 GiB` at `step=2048, ctx=131K`). Setting `--prefill-step-size 512` cuts transient mask memory 4× to `6.1 GiB` while reaching **81–85%** of practical prefill MFU (`590.5 / 547.4 tok/s`). Pairing the 4-bit DSpark drafter at `--draft-block-size 5` (avoiding `block_size=8` expert-union bloat) reaches **`54.3 tok/s` on Python code**, with a break-even acceptance of `a* = 2.48` tokens/round.

---

## Quickstart

Install the `mlx-vlm` branch containing the Inkling-2 serving fastpaths:

```bash
uv tool install "git+https://github.com/existential-birds/mlx-vlm.git@perf/inkling2-serving" --force
```

Launch the qualified OpenAI-compatible server configuration (`best-config.json`):

```bash
mlx_vlm.server \
  --model existentialbirds/Inkling-2-Mini-4bit \
  --prefill-step-size 512 \
  --cache-limit 4294967296 \
  --max-num-seqs 4 \
  --port 8080
```

For code generation or greedy/low-temperature single-stream tasks, optionally enable the 4-bit DSpark drafter (`--draft-block-size 5`):

```bash
mlx_vlm.server \
  --model existentialbirds/Inkling-2-Mini-4bit \
  --draft-model existentialbirds/Inkling-2-Mini-MTP-4bit \
  --draft-kind dflash \
  --draft-block-size 5 \
  --prefill-step-size 512 \
  --port 8080
```

Re-run the benchmark harness or long-context needle gate locally:

```bash
python3 skills/inference-optimisation/scripts/bench.py best-config.json
python3 skills/inference-optimisation/scripts/bench.py best-config.json --long-context 131072
```

---

## Repository Layout

| Path | Description |
|---|---|
| [`PLAN.md`](PLAN.md) | Chronological experiment log: hypotheses, interleaved A/B numbers, profiling breakdowns, and rejected levers |
| [`THEORY.md`](THEORY.md) | Measured M4 Max hardware figures and analytical byte/FLOP ceilings |
| [`PROPOSAL.md`](PROPOSAL.md) | Initial architecture survey, open research questions, and phased test plan |
| [`best-config.json`](best-config.json) | Qualified winning server configuration and pinned artifact revisions |
| [`ledger.jsonl`](ledger.jsonl) | Append-only JSONL record of every benchmark and qualification run |
| [`configs/`](configs/) | Individual experiment configs (`e0`–`e4`) and `ceiling.py` model specs |
| [`scripts/`](scripts/) | Detached MLX launcher, macOS `proc_pid_rusage` memory probe, and Metal microbenchmarks |
| [`skills/inference-optimisation/`](skills/inference-optimisation/) | Vendored benchmarking methodology, `bench.py` harness, `ceiling.py`, and MLX reference notes |
| [`reference/llama-cpp/`](reference/llama-cpp/) | Reference `inkling2-loop` implementation and GGUF conversion mapping from `llama.cpp` |

## Hardware Testbed

MacBook Pro (`Mac16,6`), Apple M4 Max (40-core GPU), 48 GB unified memory, macOS 26.6.1:
- **440 GB/s** sustained Metal device copy (`438–454 GB/s` across reps)
- **15.2 TFLOPS** FP16 GEMM (`4096³` / `8192³`)
- All runs executed on AC power under a **36 GiB** `ri_phys_footprint` cap.

## License

- **Repository code, configs, and documentation:** [MIT License](LICENSE), matching the underlying [`inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill), [`llama.cpp`](https://github.com/ggml-org/llama.cpp), and [`mlx-vlm`](https://github.com/Neywa-Labs/mlx-vlm) projects (third-party copyright notices in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)).
- **Published `Inkling-2-Mini` checkpoints on Hugging Face:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) and the Thinking Machines **[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)**, inherited from [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
