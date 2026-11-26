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
