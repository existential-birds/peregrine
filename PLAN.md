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

## Model selection (2026-11-25)

User opened the target to other labs (Ornith, Thinking Machines, Reflection,
American open labs), chasing higher tok/s. Ceilings from
`configs/ceiling-*-m4max.json`; quality from Terminal-Bench 2.1 leaderboard
(rank/value) where present:

| Model | Lab | TB 2.1 | MLX | c1 ceiling @4K | + draft | c4 agg @4K |
|---|---|---|---|---:|---:|---:|
| Qwen3.8-27B 4bit (current) | Qwen | 73.0 | ✓ | 28 | 68 (MTP k2) | 429 |
| **Ornith-1.5-35B-A3B MLX-4bit + DFlash** | Ornith | n/a (1.0-35B: 64.2) | ✓ first-party | **206** | 254–508 | **1758–3125** |
| Ornith-1.5-9B MLX-4bit + DFlash | Ornith | 47.0 | ✓ first-party | ~86 | ~150–250 | ~700 |
| Inkling-2-Mini (26B MoE, loop×3) | TML | **71.6** | ✗ custom loop arch | ~145 eff. | — | — |
| OLMoE-1B-7B 4bit | Ai2 | not listed | ✗ no quant | 453 (154 @32K) | — | 2475 |
| Reflection Beam-1.0-501B-A23B | Reflection | — | ✗ 280 GB @4bit, doesn't fit | — | — | — |

**Verdict:** single-stream 1000+ tok/s is not reachable from these labs on
this machine; **1000+ aggregate at c4 is** — via Ornith-1.5-35B-A3B + DFlash.
DFlash draft is 0.77 GB, block-diffusion (parallel block proposals, README
shows block size 8), natively supported by our server
(`--draft-model … --draft-kind dflash --draft-block-size`), and Ornith ships
first-party MLX quants. It is also the exact model family the skill's B390
recipe hill-climbed (local-ai-registry PR #453) — a published recipe to mine.
It is a Qwen-family reasoning model (think-blocks, `reasoning_content`), so
the harness needs no changes.

## STATUS

Target switched (user decision, 2026-11-25): **Inkling-2-Mini on MLX** —
novelty over performance; ~145 tok/s ceiling accepted. Implementation plan:
`PROPOSAL.md` (build `inkling2_loop` for mlx-vlm + affine 4-bit conversion,
validated against a llama.cpp oracle). Ornith-1.5-35B-A3B + DFlash remains
the documented fallback/comparator. Baseline experiments deferred until the
MLX port lands; Phase 0 (llama.cpp oracle baseline) can run meanwhile.
