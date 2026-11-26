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

Target (user decision, 2026-11-25): **Inkling-2-Mini on MLX** — novelty over
performance; ~145 tok/s ceiling accepted. **No port needed:** `inkling2_loop`
is already implemented inside `mlx_vlm/models/inkling2/` (routed at
`utils.py:82`; controllers/bank/message-queries in `language.py`); the fork
**`existential-birds/mlx-vlm`** (`main` @ `e8035ae3`, upstream
`Neywa-Labs/mlx-vlm`) adds Mini-specific kernels absent from PyPI 0.8.1.
Remaining plan: `PROPOSAL.md` v2 — smoke-test serving from the fork, pick
quant (NVFP4 as-is vs affine 4-bit convert), validation gates, hill-climb.
Ornith-1.5-35B-A3B + DFlash remains the documented fallback/comparator.

---

## Inkling-2-Mini Log

### 2026-11-25 20:45 — Preflight & Phase 1 smoke test (`thinkingmachines/Inkling-2-Mini` @ `c9d0283`)

- **Runtime:** editable install of fork `existential-birds/mlx-vlm` @ `e8035ae3`
  (`mlx 0.32.5`, Python 3.14 uv tool env). AC power verified.
- **Checkpoint discovery:** `thinkingmachines/Inkling-2-Mini` (`c9d0283`)
  bundles **`mtp.safetensors` (`635.8 MB`)** alongside `model-00001-of-00001.safetensors`
  (`19.5 GB`); `model.safetensors.index.json` omits the sidecar, which is why
  `PROPOSAL.md` believed no drafter weights shipped. `Inkling2DSparkSplitter`
  loads it directly (`--draft-kind dflash`, `block_size=8`).
- **Harness & launcher fixes before measurement:**
  1. `scripts/launch-mlx.sh`: replaced Linux-only `setsid` with Python
     `os.setsid() + os.execv` (preserving exact server PID in `.server.pid`)
     and added `--model` override support.
  2. `scripts/mem-macos.sh`: `footprint` printed `<N> KB` (breaking bash
     arithmetic) and suspended a 20 GB Metal task for 74 ms–29 s per probe
     (stalling decode from `38.6` down to `9.7 tok/s` when sampled). Replaced
     with `libproc.proc_pid_rusage` (`RUSAGE_INFO_V4.ri_phys_footprint`), which
     matches `footprint` within 32 KiB in `0.7 ms` with zero task suspension.
  3. `skills/inference-optimisation/scripts/bench.py` + `queue.sh`: `healthy()`
     now checks `/health` (`model_ready` and `preload_failures`) so background
     preload isn't misread as `0.8 s` startup via `/v1/models` HF cache scan;
     `queue.sh` now uses `fcntl.flock` on macOS where BSD lacks `flock(1)`.
  4. Template control: `chat_template.jinja` ignores `enable_thinking` and reads
     `reasoning_effort` (`"none"` → `Thinking effort level: 0`, default `0.9`).
- **Smoke test results (`logs/server-smoke-nvfp4.log` + isolated `ResponseGenerator`):**
  - Preload ready in **13.7 s**, `19.24 GiB` resident footprint, `1,048,576`
    context, `inkling` tool parser active.
  - Chat, code (`binary_search`), math+reasoning (`17 * 23 = 391`), tool call
    (`get_weather({"city": "Tokyo"})`), and vision (`32×32` solid red PNG) all
    clean on first boot — zero NaNs or looping.
  - Isolated `ResponseGenerator` c1 decode: **35.8–38.6 tok/s** no-draft
    (`25.2–27.4 ms/tok`); initial DSpark `block_size=8` probe hit **41.8 tok/s**
    (`256` tokens in `104` rounds, `a = 2.46`).

### 2026-11-25 21:00 — Phase 2a MoE dequant verification & 4-bit conversion

- **ModelOpt NVFP4 oracle check (`layer 0 w2`, `128 × 2304 × 1152`):**
  `mlx_vlm` dequantization (`ScaledQuantizedSwitchLinear` with `amax / 2688.0`)
  matched an independent FP32 NumPy E2M1+E4M3 decoder to
  **`max_abs = 7.45e-9`, `rel_l2 = 2.65e-8`**.
- **Kernel & precision finding:**
  - `hf_quant_config.json` stores MoE layers `0..19` as `nvfp4` (`g=16`),
    coda MoE layers `20..23` as `mxfp8` (`g=32`), and all dense attention +
    `embed` + `unembed` in BF16 (`_logits_from_norm` caches a `1.85 GB` FP32
    `lm_head`, costing `3.97 ms/tok` alone; total native read = `6.302 GB/tok`,
    `68 tok/s` @ 4K ceiling).
  - Single-token `w2` microbench (`B=1, L=1, top-8`): native `nvfp4` (`211.2 µs`)
    beats `affine` 4-bit (`239.7 µs`) and `mxfp8` (`358.9 µs`); moreover, the
    fork's `inkling2_grouped_expert_qmv` kernel (#2798) only accepts `nvfp4`/`mxfp8`
    expert scales (`uchar`, no affine biases).
  - Running `mlx_vlm.convert -q --q-bits 4 --mtp` quantizes BF16 dense
    attention + `embed` + `lm_head` to affine 4-bit (`g=64`, `5.236 bpw` overall,
    `16.08 GiB` target + `247 MiB` 4-bit DSpark drafter) while keeping all 24
    MoE layers in lossless native `nvfp4`/`mxfp8`. Decode weight read drops
    **`6.302 → 3.031 GB/tok`** (**139 tok/s** @ 4K ceiling).
  - Greedy logit equivalence (`verify_4bit_reload.py`): first-token cosine
    similarity `0.9986–0.9996`, `KL = 0.00000 nats`, `top1_match = True` across
    3 prompts, `51–73` exact greedy prefix tokens before first BF16 tie.
  - Note on `mlx-community/Inkling-2-Mini-4bit` (`9c0e296`): inspected and
    rejected — exported with `mlx-lm` (`ml-explore/mlx-lm#2086`), uses `mlx-lm`
    tensor prefixes (`model.language_model.*`), omits `processor_config.json`,
    `tiktoken/`, per-layer `nvfp4`/`mxfp8` config metadata, and drops the
    DSpark drafter.

### 2026-11-25 21:50 — Phase 2 interleaved A/B (`NVFP4` vs `4-bit`) & E0 baseline (`ledger.jsonl` #1–4)

- **Harness bug caught mid-queue:** `mlx_vlm.server`'s `GET /v1/models` calls
  `huggingface_hub.scan_cache_dir()` + `discover_models()` synchronously on
  every request; across a 39 GB HF cache this exceeded `bench.py`'s 5 s timeout,
  distorted `p2-4bit-cand` `startup_s` (`25.3 s` vs actual `5.0 s` preload), and
  saturated FastAPI's worker pool on `p2-nvfp4-ctl-b`. Updated `bench.py`
  `Server.healthy()` to return `True` directly once `/health` reports
  `model_ready: true` (`startup_s`: **9.2 s** NVFP4, **7.5 s** 4-bit).
- **Measured `ledger.jsonl` rows (`temperature=0.6`, `reasoning_effort="none"`, `reps=6`):**

| Run | `c1_decode` | `c1_all` | `c4_agg` | `prefill_4K` (3.3K) | `prefill_32K` (25.6K) | `ttft_ms` | `startup_s` | `mem_peak_gib` |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `p2-nvfp4-ctl-a` | 20.40 | `[25.65, 26.47, 20.73, 5.01, 13.07, 20.07]` | 34.70 | 312.3 | 376.0 | 251.0 | 13.2 | **33.82** |
| `p2-4bit-cand` | 22.12 | `[14.61, 14.97, 19.41, 24.84, 26.62, 28.29]` | 36.38 | 239.1 | 265.4 | 575.9 | 25.3* | **29.82** |
| `p2-nvfp4-ctl-b` | 33.11 | `[33.45, 32.77, 34.27, 36.28, 31.71, 18.46]` | 16.22 | 332.8 | 306.7 | 245.3 | 9.2 | **34.30** |
| `e0-4bit-ctl` | 24.60 | `[10.30, 7.83, 22.29, 26.91, 27.77, 31.14]` | **49.43** | 321.1 | 300.3 | **222.7** | **7.5** | **30.28** |

- **Diagnosis (critical finding for Phase 4):**
  1. `mlx_vlm.server` (`ResponseGenerator`) enters `wired_limit(model)` (~36 GiB)
     without bounding `mx.set_cache_limit(...)`. Metal's allocator retains
     `13–14 GiB` of temporary buffers across `c1` / `c4` / `32K` shapes.
  2. On a 48 GB Mac (~12–14 GB macOS + user processes), NVFP4 (`21.0 GiB`
     weights + FP32 `lm_head` + `13.3 GiB` Metal cache = **`34.30 GiB`**)
     drives the host into swap mid-run (`p2-nvfp4-ctl-b` rep 6 dropped `36.3 →
     18.5 tok/s` and `c4` collapsed to `16.22 tok/s`), and leaves compressed
     pages that depress reps 1–2 of the immediately following 4-bit run
     (`10.3, 7.83 → 31.14 tok/s`).
  3. Isolated verification with `mx.set_cache_limit(2 GiB)` (`task-307`) ran
     `models/Inkling-2-Mini-4bit` at **43.99 tok/s** decode with **`16.24 GiB`
     peak memory** (`0.16 GiB` cache) — vs `38.6 tok/s` / `21+ GiB` for NVFP4.
  4. **Verdict (pending user sign-off on 4-bit dense quant):** `models/Inkling-2-Mini-4bit`
     saves **4.0 GiB** peak footprint (`30.28` vs `34.30 GiB`, mandatory to keep
     131K + soak under the 36 GiB cap), boots faster (`7.5` vs `9.2 s`), triples
     post-prefill `c4` stability (`49.43` vs `16.22 tok/s`), and doubles the
     theoretical decode ceiling (`139` vs `68 tok/s`). Next lever #1 in Phase 4
     must bound Metal buffer cache (`mx.set_cache_limit`) to eliminate swap
     noise before sweeping DSpark and server flags.
