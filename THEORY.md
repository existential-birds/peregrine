# THEORY — Qwen3.8-27B 4-bit MLX on M4 Max (48 GB)

From `configs/ceiling-qwen38-27b-4bit-m4max.json` via the skill's
`ceiling.py`. Hardware measured 2026-11-25 (`scripts/measure_metal.py`):
device copy **440 GB/s** (438–454 across reps; conservative value used),
fp16 GEMM peak **15.2 TFLOPS** (no thermal drop 4096³ → 8192³), MFU 0.5 for
prefill.

## Bytes per token (decode, from safetensors headers)

| Component | GB/token |
|---|---:|
| Dense weights (model) | 14.615 |
| LM head (4-bit) | 0.715 |
| Embed | excluded (gather of current row only) |
| MTP draft (k=1, own head = 0, shared) | 0.239 |
| KV at ctx C | 65,536 B × C (16 attn layers × 2 × 4 heads × 256 × fp16) |

GDN recurrent state (48 layers, fixed per stream) is **not** counted —
estimated ~0.1–0.15 GB/token read+write, a ~1% correction, exact shape
unverified.

## Decode ceilings (aggregate tok/s)

| Drafts | accept | 4K | 32K | 128K | 256K |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 28 | 25 | 18 | 14 |
| 1 | 1.80 | 50 | 45 | 33 | 24 |
| 2 | 2.50 | 68 | 61 | 45 | 33 |
| 3 | 3.00 | 81 | 73 | 54 | 40 |

Concurrency 4 (weights read once per step for the batch): no-draft 429/294/
142/84 across 4K→256K; MTP k=2 scales to 961/681/341/205.

**Acceptance values are priors, not measurements.** MTP block size is 3, so
k > 3 is impossible. If real acceptance at depth 2–3 is weak (Ornith
precedent: ~14% at depth 2), the reachable row is k=1 at ~50 tok/s.

## Prefill ceilings (practical, 50% of 15.2 TFLOPS)

4K: **153** · 32K: **137** · 128K: **102** · 256K: **76** tok/s
(FLOPs/token ≈ 2 × 24.5B active linear params + causal attention term.)

## KV and memory

- fp16 KV: 64 KiB/token → **17.18 GB per 256K stream**, 8.59 GB at 131K.
- 131K needle gate at fp16 KV: 16.05 (weights) + 8.59 (KV) + state + runtime
  ≈ 27–29 GiB — fits the 36 GiB cap. Concurrent 2 × 131K soak needs KV quant
  or a raised wired limit.
- Wired limit is at system default (`iogpu.wired_limit_mb = 0`); raising it
  requires sudo, must be recorded and restored.

## Startup floor

16.05 GB weights from disk or page cache + Metal kernel compilation (first
boot). No persistent kernel cache across restarts is assumed — verify.

---

# THEORY — Inkling-2-Mini on M4 Max (48 GB)

From [ceiling-inkling2-mini-4bit-m4max.json](configs/ceiling-inkling2-mini-4bit-m4max.json)
via `skills/inference-optimisation/scripts/ceiling.py` and safetensors header
accounting (`thinkingmachines/Inkling-2-Mini` @ `c9d0283` converted with
`mlx_vlm.convert -q --q-bits 4 --mtp`).

## Architecture & loop accounting

- **Physical layers:** 24 (`0..5` prelude, `6..17` core × 3 passes, `18..23` coda)
  → **48 layer visits per token**.
- **MoE per layer:** 128 routed experts, top-8 active (`6.25%` per single-token
  visit), 0 shared experts, `hidden=2304`, `intermediate=1152`.
  - Native checkpoint (`hf_quant_config.json`): layers `0..19` experts are
    `nvfp4` (`group_size=16`, `0.5625 B/param`), layers `20..23` (coda) are
    `mxfp8` (`group_size=32`, `~1.03 B/param`), and all dense attention /
    `embed` / `unembed` / `vision_tower` weights are BF16/FP32 (`_logits_from_norm`
    caches a `1.85 GB` FP32 `lm_head` copy on decode).
  - Converted 4-bit (`models/Inkling-2-Mini-4bit`, `17.266 GB` / `16.08 GiB`
    stored): quantizes BF16 dense attention + `embed` + `lm_head` to affine
    4-bit (`group_size=64`) while preserving native `nvfp4`/`mxfp8` experts
    (keeping zero double-quantization loss and keeping the fork's custom
    `grouped_expert_projections` kernel active).
- **KV layout across 48 visits (`make_cache`):**
  - 40 sliding-window visits (`AttentionSinkKVCache(1024)`) → bounded at
    `40 × 1024 × 4,096 B = 167.8 MB` total above 1K context.
  - **8 unbounded global visits** (physical layers `5` ×1, `11` ×3, `17` ×3,
    `23` ×1; `8 KV heads × 128 dim × 2 (K,V) × 2 B = 4,096 B/visit/tok`) →
    **`32,768 B/token` (`32 KiB/token`)** in bf16 (`4.30 GB` / `4.00 GiB` at
    131K; `8.59 GB` at 256K).

## Bytes per token (decode, loop-weighted from safetensors headers)

| Component | Native NVFP4 (`GB/tok`) | 4-bit dense + NVFP4/MXFP8 MoE (`GB/tok`) |
|---|---:|---:|
| Dense text weights (48 visits) | 2.610 | 0.931 |
| Active routed experts (top-8/128 across 48 visits) | 1.839 | 1.839 |
| LM head (`unembed`, 201,024 × 2,304) | 1.853 (FP32 cached) | 0.261 (4-bit) |
| **Total weight read (0 ctx, no draft)** | **6.302** | **3.031** |
| KV read at ctx `C` (8 global slots) | `32,768 B × C` | `32,768 B × C` |
| DSpark draft (`mtp.safetensors`, 3 layers + folded Markov head) | 0.739 + 0.926 head | 0.363 + 0.261 head |

## Ceilings (4-bit dense + native MoE, 440 GB/s, 15.2 TFLOPS @ 50% MFU)

- **c1 decode (no draft):** **145 tok/s** @ 0K · **139** @ 4K · **107** @ 32K · **60** @ 128K · **38** @ 256K
  (Native NVFP4 no-draft ceiling @ 4K is **68 tok/s** due to `6.30 GB/tok`.)
- **c4 aggregate decode (no draft):** **836** @ 4K · **578** @ 32K · **281** @ 128K · **167** @ 256K
- **DSpark block-diffusion ceiling:** single-pass draft (`0.624 GB` incl. 4-bit
  target `lm_head`), but `L`-token verify increases routed-expert union
  `1 - (120/128)^L` across `29.43 GB` loop-weighted experts; shallower
  `draft_block_size` (3–5) trades peak `accept` against expert-union bloat.
- **Prefill (practical, 50% MFU):** **727** @ 4K · **641** @ 32K · **455** @ 128K · **328** @ 256K tok/s.
- **Memory under 36 GiB cap:** `16.08 GiB` weights + `0.16 GiB` sliding KV +
  `4.00 GiB` global KV @ 131K + runtime ≈ **21–23 GiB** single-stream (and
  **~25–28 GiB** for `2 × 131K` concurrent soak — fits under 36 GiB without
  touching `iogpu.wired_limit_mb`).
