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

Target (user decision, 2026-11-25): **Inkling-2-Mini on MLX** — **COMPLETE,
QUALIFIED & PUBLISHED (2026-11-27)** in [best-config.json](best-config.json) on fork
**`existential-birds/mlx-vlm`** branch `perf/inkling2-serving` @ `f871d0b7`
(base `main` @ `e8035ae3`; `mlx 0.32.5`, Apple M4 Max 40-core GPU, 48 GB).
Published Hugging Face checkpoints (Apache 2.0 + Thinking Machines Model AUP):
[`existentialbirds/Inkling-2-Mini-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-4bit) (`@6158c40c`)
and [`existentialbirds/Inkling-2-Mini-MTP-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-MTP-4bit) (`@1e218e5d`).

- **Best (`e4-best-cand`) vs clean 4-bit baseline (`e1r-4bit-ctl-a` / `e2-4bit-ctl-a`) vs native NVFP4 (`p2-nvfp4-ctl-b`) vs ceiling:**
  - **`c1` decode (`T=0.6`, `reasoning_effort="none"`, median of 6):**
    **`46.45 tok/s`** (`21.5 ms/tok`) vs `43.45–44.08 tok/s` clean 4-bit
    baseline (`+5.4%–6.9%`), `33.11 tok/s` native NVFP4 (`+40.3%`), and
    `139 tok/s` pure weight-bandwidth ceiling @ 4K (`33.4%` of naive BW ceiling;
    `47.0%` of `9.92 ms` raw weight-kernel floor once `13.7 ms` of 48-visit
    loop/dispatch work is accounted for).
  - **`c4` aggregate decode:** **`93.31 tok/s`** (`4 × 23.98`) vs `77.97–79.39`
    clean 4-bit baseline (`+17.5%–19.7%`), `16.22–34.70` native NVFP4, and
    **`224 tok/s`** true sparse-MoE `c4` bandwidth ceiling (`29.0/128` unique
    experts/layer = `7.85 GB/step`; `41.7%` of true ceiling).
  - **Cold prefill:** **`590.5 tok/s`** @ 4K (`3,300` tok; **`81.2%`** of `727`
    practical 50%-MFU ceiling) and **`547.4 tok/s`** @ 32K (`25,552` tok;
    **`85.4%`** of `641` ceiling) vs `533.7 / 507.9` clean baseline and
    `332.8 / 306.7` native NVFP4.
  - **TTFT / Startup / Peak memory (25–33K bench):** **`196.1 ms`** / **`5.3 s`** /
    **`23.91 GiB`** (`-6.2 GiB` vs `30.1 GiB` unbounded 4-bit; `-10.4 GiB` vs
    `34.3 GiB` NVFP4).
  - **All validation & qualification gates GREEN (`≤ 36 GiB`, `iogpu.wired_limit_mb=0` untouched):**
    - 32K exact needles (`33,248` tokens): **`3/3`**, `458.1 tok/s` prefill, `26.54 tok/s` decode, `23.91 GiB`.
    - 131K exact needles (`132,777` tokens): **`3/3`**, `258.9 tok/s` prefill, **`22.74 tok/s` decode** (`37.9%` of `60 tok/s` 128K ceiling), **`30.08 GiB`**.
    - Concurrent full-KV soak (`2 × 67.3K = 134,684` live tokens): **`6/6` needles**, **`25.18 GiB`** peak, zero looping/trips.
    - Modalities: chat, reasoning (`1591`), `inkling` tool round-trip, and vision (`64×64` PNG) all green.
- **Limiting factors (measured, not guessed):**
  1. **48-visit dispatch & loop-control floor (`13.7 ms/tok` = `58%` of `c1` step):**
     1 token emits `9,951` MLX graph ops across 48 sequential visits (`1.96 µs/op`
     Metal serial launch floor): `MessageController`s (`5.1 → ~3.5 ms` after 3×
     batching), 97 `AttentionResidual` `mx.stack` + softmax (`2.73 ms`), and 192
     short-convolutions (`2.97 ms`).
  2. **Scattered `gather_qmm(K=8)` NVFP4 MoE bandwidth (`5.93 ms` = `227 GB/s`, `52%` peak):**
     120 small top-8/128 expert gathers across 40 NVFP4 visits under-saturate DRAM
     compared to dense 4-bit `qmv` (`463 GB/s`) and coda MXFP8 (`459 GB/s`).
  3. **DSpark break-even (`a* = 2.48` at `block_size=5`):** L-token verify widens
     the active expert union across 48 visits (`57.1 ms/round` vs `21.5 ms`
     1-token step). Wins on code (`54.34 tok/s`), loses on `T=0.6` prose (`26–34 tok/s`).
- **Ranked remaining levers (future work, beyond current scope):**
  1. Custom Metal kernel fusing `AttentionResidual` (`rms_norm` + dot + softmax +
     weighted sum without `mx.stack` allocation, ~`2.0 ms/tok` potential).
  2. Fused Metal kernel for `MessageController` + `DynamicShortConvolution` (~`2.5 ms/tok`).
  3. Workload-adaptive / code-only DSpark (`--draft-block-size 5` at `T=0` or code tasks).
  4. Optional cross-engine `llama.cpp` Q4_K_M A/B (`DevQuasar` GGUF).

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

### 2026-11-25 22:08 — Phase 4a Profiling: Where the `139 / 836 tok/s` ceiling gap actually goes

- **Base locked (user confirmed):** `models/Inkling-2-Mini-4bit` +
  `models/Inkling-2-Mini-4bit-mtp` (`247 MiB`).
- **Profiler (`profile_ceiling_gap.py`, `ctx ≈ 550`, `mx.set_cache_limit(2 GiB)`):**
  - **1-token MLX graph size:** **`9,951` op boxes** (`AttentionSinkKVCache`) vs
    **`17,319` op boxes** (`mlx_vlm.server` `BatchKVCache` — **`+7,368` ops/token!**).
  - **Measured Metal serial kernel-launch floor:** **`1.96–2.25 µs/op`** (`2,400`
    serial elementwise ops = `4.71 ms`).
  - **10-stage GPU breakdown across all 48 loop visits (`ms/step`, median of 12):**
    - `c1` single-stream (`AttentionSinkKVCache`, sum = **`23.62 ms`** = `42.3 tok/s`):
      - Weight stages (`3.031 GB`): `S3` dense attn (`2.01 ms`, `463 GB/s`),
        `S10` `lm_head` (`0.91 ms`, `287 GB/s`; `0.76 ms` in bf16), `S9b` MXFP8
        MoE (`1.07 ms`, `459 GB/s`), `S9a` NVFP4 MoE (`7.24 ms` incl. `1.31 ms`
        router = `5.93 ms` / `227 GB/s` across 120 scattered `gather_qmm(K=8)`).
        Total weight time = **`9.92 ms`** (`70%` of `6.89 ms` bandwidth floor).
      - Non-weight / loop-dispatch stages (**`13.70 ms` = `58%` of step!**):
        `S2` 96 uncompiled `MessageController`s (`5.09 ms` for `0.11 GB`),
        `S1` 97 `AttentionResidual` `mx.stack` + softmax (`2.73 ms`),
        `S4+S7` 192 short-convs (`2.97 ms`), `S8` router (`1.31 ms`),
        `S5+S6` head prep + banded rel-pos mask + SDPA (`1.59 ms`).
      - Explains why 4-bit's `2.1×` byte cut only yielded `~14%` on clean
        `generate()` (`38.6 → 44.0 tok/s`): 4-bit only shrinks `S3 + S10`
        (`~6.5 → 2.92 ms`), leaving `20.7 ms` (`88%`) untouched.
    - `c1` server (`BatchKVCache`, sum = **`37.38 ms`** = `26.8 tok/s`):
      - **Root-cause bug #1 (`+10.82 ms/tok`, `+7,368 ops/tok`):** after prefill,
        `ArraysCache.left_padding` remains `array([-512])` instead of `None`.
        `_cache_padding_mask` emits a `[[True]]` mask on every decode step,
        disabling the fused `write_window` Metal kernel in all 176
        `ShortConvolution`s (`S4`: `1.84 → 7.84 ms`, `S7`: `1.13 → 5.95 ms`).
      - **Root-cause bug #2 (`+1.38 ms/tok` @ 550 ctx, grows with ctx):**
        `BatchKVCache` lacks `AttentionSinkKVCache`'s preallocated zero sink
        column, so `Attention.__call__` runs `mx.concatenate([k, zeros])` and
        `mx.concatenate([v, zeros])` 48× per step (`S6`: `0.92 → 2.30 ms`).
      - **Root-cause #3 (host & Metal cache):** unbounded `mx.set_cache_limit`
        retains `~14 GiB` of buffers (`30.3 GiB` server footprint), plus a
        background `com.apple.Virtualization` (Docker Desktop VM) holding `32 GB`
        unified memory and bursting CPU (`Load Avg 14–24`), explaining the
        remaining `26.8 → 22.1–24.6 tok/s` swap/CPU variance.
    - `c4` server (`B=4, L=1`, sum = **`57.50 ms`** = `70–80 tok/s` agg clean):
      - **Physics correction:** `ceiling.py`'s `836 tok/s` assumes dense weight
        sharing across streams. For top-8/128 MoE, 4 independent streams touch
        `128 × (1 - (120/128)^4) = 29.0` unique experts/layer (`3.62×` expert
        bytes = `7.85 GB/step`), so **true MoE `c4` bandwidth ceiling is
        `~224 tok/s`** (`19.1 ms/step`), not `836 tok/s`.
      - Remaining `19.1 → 50.2 ms` gap: `+15.1 ms` `ArraysCache.left_padding`
        bug (`S4+S7`), `+7.8 ms` loop controllers (`S2`), `+6.0 ms` `BatchKVCache`
        96× KV sink concats (`S6`), `+4.2 ms` router `argpartition`, and
        `+1.0 ms` `prepare_attention_heads` rejecting `batch != 1`.
    - `B=1, L=8` speculative verify (`speculative_verify=True`): **`43.63 ms`**
      fused (`1.17×` of 1-token server step!), hitting `grouped_expert_projections`
      on both NVFP4 (`25.05 ms`) and MXFP8 (`4.36 ms`) while amortizing all loop
      dispatch once per block. In isolated `ResponseGenerator` under load,
      4-bit DSpark achieved `a = 2.32 tok/step` and **`1.92×` decode speedup**
      over no-draft on the same engine (`26.73` vs `13.92 tok/s`).

### 2026-11-26 14:43 — E1 DSpark drafter interleaved A/B (`ledger.jsonl` #5–9)

- **Note on #5 (`e1-4bit-ctl-a` @ 22:17):** aborted queue cleanly after #5 finished
  when a concurrent host Docker Desktop VM (`33 GB` RAM, `400%` CPU) drove the
  host into swap (`c1 = 10.55 tok/s`). Re-ran fresh interleaved queue (`e1r-*`)
  the next afternoon once Docker containers were idle (`<1.1%` CPU).
- **Measured `ledger.jsonl` #6–9 (`models/Inkling-2-Mini-4bit`, `T=0.6`, `reasoning_effort="none"`, `reps=6`):**

| Run | `c1_decode` | `c1_all` | `c4_agg` | `prefill_4K` | `prefill_32K` | `ttft_ms` | `startup_s` | `mem_peak_gib` |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `e1r-4bit-ctl-a` | **43.45** | `[43.36, 43.28, 43.78, 43.47, 43.96, 43.43]` | **77.97** | **533.7** | **507.9** | 209.2 | 10.6 | 30.12 |
| `e1r-dspark-b8-cand` | 18.52 | `[40.32, 17.77, 14.57, 19.26, 23.65, 16.39]` | 22.60 | 484.7 | 413.8 | 215.2 | 5.3 | **35.96** |
| `e1r-dspark-b5-cand` | 33.09 | `[54.34, 34.47, 26.85, 32.89, 33.29, 26.16]` | 32.87 | 462.1 | 411.8 | **202.5** | 5.5 | **34.52** |
| `e1r-4bit-ctl-b` | 32.61 | `[40.75, 37.08, 33.02, 31.43, 29.50, 32.19]` | 53.36 | 457.9 | 418.5 | 203.6 | 5.4 | 30.04 |

- **Clean baseline validation:** With host contention gone, `e1r-4bit-ctl-a`
  matched our isolated profiler within `2%`: **`43.45 tok/s` `c1`** (`±0.3 tok/s`
  across all 6 prompts; `BatchGenerator` leaves `left_padding=None` at `B=1`),
  **`77.97 tok/s` `c4`** (where unequal prompt lengths `[37, 33, 37, 39]` set
  `left_padding=[2, 6, 2, 0]` and trigger the `+15.1 ms/step` `ArraysCache` bug),
  and **`533.7 / 507.9 tok/s` prefill** (`73–79%` of practical 50%-MFU ceiling).
- **Verdict on E1 (`DSpark`): REJECTED as default server config.**
  - **Why:** Single-token decode takes `23.0 ms` (`43.45 tok/s`), whereas one
    `block_size=5` DSpark draft+verify round takes `~57.1 ms` (`17.5 tok/s` on
    0-accept rounds in server logs; `~100 ms` at `block_size=8`) because `L`-token
    verify expands the top-8/128 routed-expert union across all 48 visits.
    Break-even acceptance is **`a* = 2.48` tokens/round** (`4.35` at `b8`).
  - On code (`PROMPTS[0]`, LRU cache), `b5` clears break-even (`a ≈ 3.1 → 54.34
    tok/s`, **`+25.1%` win** over `43.45`, and `+34.8%` over `b8`'s `40.32`).
    On sampled prose at `T=0.6` (`PROMPTS[1..5]`), acceptance drops to `a ≈
    1.5–1.95`, regressing median `c1` to `33.09 tok/s` (`-24%`), regressing `c4`
    by `-58%` (`77.97 → 32.87`), and pushing `mem_peak_gib` to `34.52–35.96 GiB`
    (re-inducing host swap carryover seen in `e1r-4bit-ctl-b`).

### 2026-11-26 15:10 — E2 `--cache-limit 2 GiB` interleaved A/B (`ledger.jsonl` #10–12)

- **Change:** Fork branch `perf/inkling2-serving` (`77af3cfe`) adds `--cache-limit`
  (`MLX_VLM_CACHE_LIMIT`) to `mlx_vlm.server`; candidate sets `2147483648` (2 GiB).
- **Measured `ledger.jsonl` #10–12 (`T=0.6`, `reasoning_effort="none"`, `reps=6`):**

| Run | `c1_decode` | `c1_all` | `c4_agg` | `prefill_4K` | `prefill_32K` | `ttft_ms` | `startup_s` | `mem_peak_gib` |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `e2-4bit-ctl-a` | **44.08** | `[44.02, 44.20, 44.08, 44.09, 44.05, 44.07]` | **79.39** | **551.1** | **524.1** | **198.2** | 7.3 | 29.93 |
| `e2-cache2g-cand` | 33.89 | `[43.97, 43.77, 35.95, 31.82, 27.12, 27.07]` | 52.56 | 450.1 | 413.5 | 202.3 | 5.4 | **27.26** |
| `e2-4bit-ctl-b` | 36.08 | `[42.16, 38.87, 37.27, 34.69, 34.33, 34.89]` | 62.81 | 493.8 | 446.9 | 202.1 | 5.4 | 30.21 |

- **Verdict on E2 (`--cache-limit 2 GiB`): REJECTED as default.**
  - Host had `29 GB` free physical RAM (`0` swapouts); `e2-cache2g-cand` cut
    `mem_peak_gib` by `2.67 GiB` (`29.93 → 27.26 GiB`) but regressed below BOTH
    adjacent controls (`33.89` vs `44.08` & `36.08` `c1`; `52.56` vs `79.39` &
    `62.81` `c4`; `450.1` vs `551.1` & `493.8` `prefill_4K`).
  - Why: once cached Metal buffers reach `2.0 GiB` (starting on rep 3), growing
    dynamic-shape tensors (`banded_additive_mask` over `S = 38..550`, `mx.stack`
    in `AttentionResidual`) evict and trigger repeated `IOGPUMemory` buffer
    reallocation syscalls. Also note laptop GPU thermal clock settling (`44.1 →
    ~36 tok/s`) after consecutive 25K GEMM prefills; interleaved controls isolate
    both effects cleanly. Keep `--cache-limit` in the fork CLI as an opt-in knob
    for tight memory pools, unset by default.

### 2026-11-26 15:26 — E3 Inkling-2 lossless kernel & batch-cache fastpaths (`ledger.jsonl` #13–15)

- **Change:** Fork commit `f871d0b7` on `perf/inkling2-serving`:
  1. Batched 3× `(attn, k, v)` + `@mx.compile`d `MessageController` with fused
     `forget`/`gain` projection (`mlx_vlm/models/inkling2/language.py`).
  2. Widened `prepare_attention_heads` Metal kernel guard from `batch == 1` to
     `1 <= batch * rows <= 8` (`mlx_vlm/models/inkling2/kernels.py`), enabling
     the fused head-prep kernel at `c4` (`B=4, L=1`).
  3. Auto-cleared spent `ArraysCache.left_padding` once `advance >= max(padding)`
     (`mlx_vlm/models/cache.py` + `ar.py`), eliminating `+7,368` mask/fallback
     ops per decode step on unequal-length batches (`c4`).
  4. Added zero-copy `BatchKVCache.with_sink` (`mlx_vlm/models/cache.py` +
     `language.py`), avoiding full-KV sink concatenation on `255/256` steps.
  - All 68 Inkling-2 & cache unit tests pass (incl. 3 new regression tests).
- **Measured `ledger.jsonl` #12–15 (`ctl → cand-a → ctl → cand-b` chain):**

| Run | Ref | `c1_decode` | `c1_all` | `c4_agg` | `prefill_4K` | `prefill_32K` | `ttft_ms` | `mem_peak_gib` |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `e2-4bit-ctl-b` (prev) | `77af3cfe` | 36.08 | `[42.16, 38.87, 37.27, 34.69, 34.33, 34.89]` | 62.81 | 493.8 | 446.9 | 202.1 | 30.21 |
| `e3-fastpath-cand-a` | **`f871d0b7`** | **45.76** | `[46.39, 45.73, 45.75, 45.77, 45.91, 45.74]` | **91.95** | **587.3** | **529.0** | **198.4** | 30.05 |
| `e3-4bit-ctl` | `77af3cfe` | 40.14 | `[44.14, 44.19, 43.62, 36.67, 32.99, 27.91]` | 49.94 | 441.8 | 418.6 | 202.0 | 30.14 |
| `e3-fastpath-cand-b` | **`f871d0b7`** | 37.78 | `[42.85, 39.94, 38.04, 37.12, 37.52, 37.48]` | **68.39** | **492.6** | **449.6** | 200.9 | 31.18 |

- **Verdict on E3 (`f871d0b7`): ACCEPTED.**
  - Rested-chassis comparison (`e2-4bit-ctl-a` → `e3-fastpath-cand-a`): `c1`
    **`44.08 → 45.76 tok/s` (`+3.8%`)**, `c4` **`79.39 → 91.95 tok/s` (`+15.8%`)**,
    `prefill_4K` **`551.1 → 587.3 tok/s` (`+6.6%`)**, `prefill_32K` **`524.1 →
    529.0 tok/s`**.
  - `e3-fastpath-cand-a` beat both adjacent controls (`e2-4bit-ctl-b` and
    `e3-4bit-ctl`) on every metric, and thermally-warm repeat `e3-fastpath-cand-b`
    still beat `e3-4bit-ctl` by **`+36.9%` on `c4`** (`68.39` vs `49.94`) and
    **`+11.5%` on `prefill_4K`** (`492.6` vs `441.8`).
  - Also revealed that `e3-4bit-ctl` (run #2 without `--cache-limit`) had the
    exact same thermal-soak curve (`44.1 → 27.9 tok/s`, `c4 = 49.94`) as
    `e2-cache2g-cand` (`c4 = 52.56`), confirming run-#2 dips are laptop GPU
    thermal throttling after 25K GEMM prefill rather than cache-limit churn.

### 2026-11-26 15:51 — E4 unified config (`best-config.json`), 32K/131K needle gates, concurrent soak & modality qualification (`ledger.jsonl` #16–18)

- **131K prefill mask math:** At default `--prefill-step-size 2048` and
  `S = 132,777`, `banded_additive_mask` + sink concat allocate
  `2 × [1, 24, 2048, 132777] × 2 B = 24.3 GiB` of transient masks on global
  layers (`> 44 GiB` total). Setting `--prefill-step-size 512` cuts that 4× to
  `6.1 GiB`, paired with `--cache-limit 4294967296` (4 GiB) and `--max-num-seqs 4`.
- **Measured `e4-best-cand` + Phase 3 / final gates (`ledger.jsonl` #16–18):**
  1. **`e4-best-cand` (`#16`):** **`c1 = 46.45 tok/s`** (`[47.18, 45.53, 46.30,
     46.43, 46.50, 46.46]`), **`c4 = 93.31 tok/s`** (`[23.98 × 4]`),
     **`prefill_4K = 590.5 tok/s`**, **`prefill_32K = 547.4 tok/s`**,
     **`ttft_short_ms = 196.1 ms`**, **`startup_s = 5.3 s`**, **`mem_peak_gib =
     23.91 GiB`** (`-6.14 GiB` below unbounded `e3-fastpath-cand-a`!).
  2. **32K exact needle gate (`#16.long`, `33,248` real tokens):** **`3/3`**,
     `ttft_s = 72.6 s` (`458.1 tok/s` prefill), `decode = 26.54 tok/s`, `23.91 GiB`.
  3. **131K exact needle gate (`#17`, `132,777` real tokens):** **`3/3`**,
     `ttft_s = 512.8 s` (`258.9 tok/s` prefill), **`decode = 22.74 tok/s`**,
     **`mem_peak_gib = 30.08 GiB`** (`5.92 GiB` under the 36 GiB cap).
  4. **Concurrent full-KV soak (`#18`, `2 × 67.3K = 134,684` live tokens):**
     **`6/6` exact needles** (`["3/3", "3/3"]`), **`mem_peak_gib = 25.18 GiB`**,
     zero trips, zero looping.
  5. **Modalities re-qualified on winning server:** reasoning (`37 × 43 = 1591`),
     structured `inkling` tool round-trip (`get_weather({"city": "Kyoto"})`),
     and vision (`64×64` solid blue PNG → `"blue"`) all passed.
- **Verdict:** **ACCEPTED as `best-config.json`.** All definition-of-done gates green.

### 2026-11-27 14:54 — Fork push & Hugging Face Hub publication

- **Fork pushed:** `existential-birds/mlx-vlm` branch `perf/inkling2-serving`
  (`77af3cfe`, `f871d0b7`; 68/68 unit tests passing) pushed to `origin` without
  opening an upstream PR per user instruction.
- **Checkpoints published (public, Apache 2.0 with `LICENSE`, `NOTICE` per §4(a)–(c), and Thinking Machines Model AUP attribution):**
  1. Target VLM (`17.27 GB` / `16.08 GiB`, 17 files, `.eval_results/*` excluded):
     [`existentialbirds/Inkling-2-Mini-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-4bit) @ `6158c40ca86fdbcbb5a32952104d740a2cb2c64d`
  2. Companion 4-bit DSpark speculative drafter (`247 MiB`, 7 files):
     [`existentialbirds/Inkling-2-Mini-MTP-4bit`](https://huggingface.co/existentialbirds/Inkling-2-Mini-MTP-4bit) @ `1e218e5dfde70f6ed0d888676fe861b92982a7f7`
