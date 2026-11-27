# Peregrine

Apple Silicon LLM inference-optimisation harness built around measured Metal ceilings and interleaved A/B benchmarks (`ledger.jsonl`).

## Quick Start

Serve [`existentialbirds/Inkling-2-Mini-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-4bit) (`46.5 tok/s` single-stream, `93.3 tok/s` `c4`, `23.9 GiB` peak on 48 GB Apple Silicon):

```bash
uv tool install "git+https://github.com/existential-birds/mlx-vlm.git@perf/inkling2-serving" --force

mlx_vlm.server \
  --model existentialbirds/Inkling-2-Mini-4bit \
  --prefill-step-size 512 \
  --cache-limit 4294967296 \
  --max-num-seqs 4 \
  --port 8080
```

For code generation (`54.3 tok/s` `c1`), add the 4-bit DSpark drafter ([`existentialbirds/Inkling-2-Mini-MTP-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-MTP-4bit)):

```bash
--draft-model existentialbirds/Inkling-2-Mini-MTP-4bit --draft-kind dflash --draft-block-size 5
```

## Copy-Paste for Coding Agents

Paste into Claude Code, Codex, Cursor, or Antigravity to benchmark or hill-climb a model on your Mac:

```text
Clone https://github.com/existential-birds/peregrine and read skills/inference-optimisation/SKILL.md and skills/inference-optimisation/references/mlx.md before acting.
Measure this Mac's Metal bandwidth and FP16 TFLOPS (`python3 scripts/measure_metal.py`), compute ceilings (`skills/inference-optimisation/scripts/ceiling.py`), and hill-climb local serving of existentialbirds/Inkling-2-Mini-4bit (or replace with my target model) under a safe unified-memory cap (`scripts/mem-macos.sh`, AC power only).
Change one variable per experiment, interleave controls (`skills/inference-optimisation/scripts/queue.sh` + `bench.py`, medians of >=6 in `ledger.jsonl`), log every win and failure in `PLAN.md`, gate on long-context needle retrieval, and write the qualified winner to `best-config.json`.
```

## Benchmark: `Inkling-2-Mini` on Apple M4 Max (40c GPU, 48 GB)

Measured at **440 GB/s** Metal copy / **15.2 TFLOPS** FP16 GEMM (`T=0.6`, `reasoning_effort="none"`, AC power):

| Metric | Official NVFP4 | Stock `mlx-vlm` + 4-bit | Tuned ([`best-config.json`](best-config.json)) | Practical Ceiling |
|---|---:|---:|---:|---:|
| **`c1` decode (median of 6)** | `33.1 tok/s` | `44.1 tok/s` | **`46.5 tok/s`** | `139 tok/s` (4K BW) |
| **`c4` aggregate decode** | `16.2-34.7` *(swap)* | `79.4 tok/s` | **`93.3 tok/s`** | `224 tok/s` (sparse MoE) |
| **Cold prefill (3.3K / 25.6K)** | `333 / 307 tok/s` | `551 / 524 tok/s` | **`591 / 547 tok/s`** | `727 / 641 tok/s` |
| **Short TTFT / Startup** | `245 ms / 9.2 s` | `198 ms / 7.3 s` | **`196 ms / 5.3 s`** | `n/a` |
| **Peak memory (≤33K ctx)** | `34.3 GiB` | `29.9 GiB` | **`23.9 GiB`** | `36.0 GiB` cap |

- **Gates passed:** `KL = 0.00000 nats` vs native NVFP4; **3/3** exact needles at 32K & 131K (`30.1 GiB`); **6/6** concurrent `2 x 67.3K` KV soak (`25.2 GiB`); chat, reasoning, `inkling` tool calls, and vision verified.
- **Why `46.5` vs `139 tok/s`:** 48 looped layer visits emit `9,951` MLX ops/token (`13.7 ms` dispatch/loop floor vs `9.9 ms` weight kernels); 4-stream top-8/128 MoE reads `29.0` unique experts/layer (`224 tok/s` true `c4` ceiling, not `836`). Details in [`PLAN.md`](PLAN.md) and [`THEORY.md`](THEORY.md) (raw runs in [`ledger.jsonl`](ledger.jsonl)).

## Repo Map

| Path | Purpose |
|---|---|
| [`best-config.json`](best-config.json), [`configs/`](configs/) | Qualified launch config and `ceiling.py` / experiment specs (`e0`-`e4`) |
| [`skills/inference-optimisation/`](skills/inference-optimisation/) | Methodology (`SKILL.md`), MLX notes (`references/mlx.md`), `bench.py`, `ceiling.py`, `queue.sh` |
| [`scripts/`](scripts/) | `measure_metal.py`, `mem-macos.sh` (`proc_pid_rusage` probe), `launch-mlx.sh`, `stop-mlx.sh` |
| [`PLAN.md`](PLAN.md), [`THEORY.md`](THEORY.md), [`ledger.jsonl`](ledger.jsonl) | Experiment log with ceiling math + append-only JSONL run ledger |

## License

Repository contents are [MIT](LICENSE) (vendored notices in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)). Published `Inkling-2-Mini` checkpoints on Hugging Face follow [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) and Thinking Machines' [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
