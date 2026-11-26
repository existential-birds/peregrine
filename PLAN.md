# PLAN — hill-climbing Qwen3.8-27B (MLX) on M4 Max

Working directory for the inference-optimisation skill. One timestamped entry per
experiment: config, numbers, verdict, reason. Negative results are the point.

## Targets and constraints (step 0)

- **Metrics:** c1 decode (primary), c4 decode aggregate, cold prefill at 4K/32K,
  TTFT, startup. Long-context gate at 131,072 tokens (exact-match needles).
- **Memory cap:** 36 GiB server footprint (48 GB machine; macOS keeps ~8–10 GB).
- **Quality changes allowed:** none unless labelled (e.g. KV quant is a trade to
  measure, not a default).
- **Engine:** MLX via `mlx-vlm` (`mlx_vlm.server`, OpenAI-compatible, port 8080).
- **Measurement rules:** thinking off, temperature 0.6, fixed prompt panel,
  medians of ≥6, controls interleaved (`queue.sh`), AC power only.

## Inventory (2026-11-25)

| Item | Value |
|---|---|
| Machine | MacBook Pro Mac16,6, Apple M4 Max, 40-core GPU, 48 GB unified, macOS 26.6.1 |
| Bandwidth | **438–454 GB/s** measured device copy (546 GB/s spec; theory uses 440) |
| Peak fp16 GEMM | **14.7–15.2 TFLOPS** measured 4096³/8192³, no thermal drop at 8192³ |
| Wired limit | `iogpu.wired_limit_mb = 0` (system default; untouched so far) |
| Target | `mlx-community/Qwen3.8-27B-4bit` @ `main` — 16.054 GB: model 14.615, embed 0.724, lm_head 0.715 |
| Draft | `mlx-community/Qwen3.8-27B-MTP-4bit` — 0.239 GB, MTP block 3, shares target embed/head |
| Stack | mlx 0.32.4, mlx-vlm via uv tool (mlx_vlm.server: `--draft-model/--draft-kind mtp`, `--kv-bits`, `--kv-group-size`, `--kv-quant-scheme`, `--max-kv-size`, `--max-num-seqs`, `--prefill-step-size`) |
| Architecture | 64 layers (48 GDN + 16 full attn), 4 KV heads × 256 dim, hidden 5120, FFN 17408, vocab 248,320, dense 27B, native 262,144 ctx |

Ceilings: see `THEORY.md`. Baseline target: c1 ~28 tok/s ceiling without
speculation; MTP acceptance is the swing variable.

## Experiments

### E0 — baseline (pending)

`configs/base.json`. First launch includes the 16 GB download + cold boot;
discard that startup number, then run `ctl` twice to establish the baseline.
Acceptance: none yet — measured first, everything compares to this.

Planned lever order (ceiling gap × likelihood, cheapest first):

1. **E1 MTP draft** `--draft-model mlx-community/Qwen3.8-27B-MTP-4bit
   --draft-kind mtp`, sweep `--draft-block-size` 1–3 (ceiling says up to
   2.4× at a = 1.8; draft adds only ~0.24 GB).
2. **E2 KV quant** `--kv-bits 8` (+ `--kv-quant-scheme turboquant`): halves the
   64 KiB/token KV; labelled trade, watch decode regression (llama.cpp
   16/64-layer precedent: 45 → 27 tok/s).
3. **E3 prefill** `--prefill-step-size` sweep against 4K/32K prefill.
4. **E4 concurrency** `--max-num-seqs`, c4 aggregate + GDN state slots.
5. **E5 quant format** oQ4/mxfp4 variants (bytes/token) — only if a lossy
   trade is agreed.

## STATUS

Not started — awaiting first server launch and baseline.
