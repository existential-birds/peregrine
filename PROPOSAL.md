# PROPOSAL — Inkling-2-Mini on MLX: convert, validate, hill-climb

Status: revised 2026-11-25 (v2 — supersedes "port inkling2_loop"). Companion
log: `PLAN.md`.

---

## READ ME FIRST — instructions for the next agent

You are bringing Thinking Machines' **Inkling-2-Mini** (looped-layer MoE VLM)
to a fast local serving setup on this machine, using the fork
**`existential-birds/mlx-vlm`** as the runtime. This proposal contains
everything verified so far. **§3 is verified with commands given; this
section is open.** Resolve in order:

> **Correction of an earlier draft:** v1 of this proposal believed
> `inkling2_loop` was unimplemented in MLX. Wrong — it is implemented
> *inside* `models/inkling2/` (no separate package; the routing is
> `mlx_vlm/utils.py:82: "inkling2_loop": "inkling2"`), mirroring llama.cpp's
> subclass design. Don't repeat the mistake of grepping for a directory.
> The remaining work is conversion, validation, and the hill-climb — plus
> fixing whatever the smoke test finds, in the fork.

1. **End-to-end smoke test (the real Phase 1).** Switch the tool env to the
   fork (`uv tool install --editable /Users/ka/github/existential-birds/mlx-vlm
   --force`), serve the Mini from the official NVFP4 checkpoint
   (`mlx_vlm.server --model thinkingmachines/Inkling-2-Mini --port 8080`),
   and confirm: config parses (incl. `loop_config`), tensors load
   (modelopt `w13/w2` + scales + `outer_scale_denominator`), the loop runs
   (3 passes, 48 cache slots), outputs are sane (no garbage/looping).
   Everything else depends on what this finds.
2. **NVFP4 vs affine-4-bit.** The checkpoint is 19.5 GB modelopt-packed. Two
   paths: serve NVFP4 as-is (if the load path handles it end-to-end) and/or
   `mlx_vlm.convert -hf thinkingmachines/Inkling-2-Mini --q-bits 4` →
   ~15–16 GB affine. Test one MoE layer's dequant in isolation first; A/B
   both (bytes/token vs quality gates). No mlx-community quant of the Mini
   exists yet (searched 2026-11-25) — ours could be the first.
3. **Kernel coverage of the recent Mini PRs.** Fork `main` @ `e8035ae3`
   includes Mini-specific optimizations: grouped-expert matvec (#2798),
   attention-head prep (#2796), sconv kernel, Inkling tool-call streaming
   (#2810). PyPI 0.8.1 lacks them — always run the fork, never 0.8.1, for
   numbers. Check `git log --grep=inkling -i` for anything newer after
   rebasing.
4. **Drafter weights.** `mtp_config` (block_size 8, 3 draft layers, Markov
   `rnn_hidden` head, `target_layer_ids [5,23,41,48]` — 41/48 are
   loop-expanded slots) is in the config but the checkpoint ships **zero
   mtp tensors**. `mlx_vlm/speculative/drafters/inkling2_dspark/` exists.
   Find whether TML ships a Mini drafter anywhere; if not, v1 runs without
   speculation (the ~145 tok/s ceiling already accepted that).
5. **Which layers have unbounded KV.** 20/24 layers are local (sliding
   window 1024 + rel_extent 1024 → bounded). Non-local = {5, 11, 17, 23}.
   Their KV growth sets the real KV ceiling at 131K/1M context and whether
   `--kv-bits` matters. Read the attention code in the fork
   (`models/inkling2/language.py`) to settle it.
6. **Loop semantics (for understanding, not porting).** The implementation
   is in the fork: controllers (`{attn,k,v,mlp}.{down,gain,forget}`) around
   language.py:641–690, `message_queries` (710–713), bank mixing
   (742–761), loop-aware speculative verify (770). Cross-reference with the
   vendored `reference/llama-cpp/inkling2-loop.cpp` (123 lines) and the
   HF→GGUF map `reference/llama-cpp/inkling2_conversion.py` if behavior
   looks off. `middle_cycle_cross_pass_queries` (inkling2.py:210) appears
   to be the msg-query tensor mapping.
7. **Cross-engine A/B (optional but cheap).** llama.cpp @ `9c003a9` runs
   the Mini (Q4_K_M GGUF, 17.48 GB + 0.64 mmproj). Same-day back-to-back
   against MLX per `references/methodology.md`. No longer required for
   correctness validation (the arch is the same family), but a useful
   engine-efficiency datapoint — and a debugging oracle if MLX output
   looks wrong.

Also read: `skills/inference-optimisation/SKILL.md` (the discipline this
repo follows), `PLAN.md` (experiment log + model-selection context),
`THEORY.md` (measured hardware), `references/mlx.md` (MLX levers/gotchas).

---

## 1. Decision and rationale

Goal was 1000+ tok/s; ceilings said no lab model reaches that single-stream
here. The user accepted the trade: **Inkling-2-Mini at ~145 tok/s ceiling for
architectural novelty** — looped layers, near-Qwen3.8-27B quality
(Terminal-Bench 2.1: 71.6 vs 73.0). The implementation already exists
upstream; what remains in scope for this workspace:

- convert/quantize (if needed) and serve from the fork,
- validate output quality (gates, needles),
- hill-climb serving speed under the skill's discipline, and
- carry any bug fixes in the fork (PR'd upstream).

Fallback/comparator: Ornith-1.5-35B-A3B MLX-4bit + DFlash (206–508 tok/s
ceiling, first-party MLX, dflash drafts) — analysis in `PLAN.md`.

## 2. Project context (all a future model needs)

- **Repos:** this one (`mlx-experiment`, the experiment working dir: harness,
  configs, ceilings, log) and the fork **`existential-birds/mlx-vlm`** at
  `/Users/ka/github/existential-birds/mlx-vlm` (remotes: `origin` = fork,
  `upstream` = `Neywa-Labs/mlx-vlm`, base `main` @ `e8035ae3`; read its
  `CONTRIBUTING.md` before changing it). The `uv tool`-installed mlx-vlm is
  PyPI 0.8.1 — **lags the fork; never benchmark on 0.8.1.**
- **Machine:** MacBook Pro Mac16,6, Apple M4 Max (16-core CPU, 40-core GPU),
  48 GB unified, macOS 26.6.1. Measured (2026-11-25, `scripts/measure_metal.py`):
  device copy **438–454 GB/s** (theory uses 440), fp16 GEMM **15.2 TFLOPS**
  (no thermal drop at 8192³). Wired limit at system default
  (`iogpu.wired_limit_mb=0`, untouched).
- **MLX stack:** `mlx 0.32.4` (pyenv python3.12 for measurement), mlx-vlm
  venv at `/Users/ka/.local/share/uv/tools/mlx-vlm` (python3.14). Server:
  `mlx_vlm.server` (OpenAI-compatible, port 8080; flags include
  `--draft-model --draft-kind {dflash,eagle3,mtp} --draft-block-size`,
  `--kv-bits --kv-quant-scheme {uniform,turboquant}`, `--max-kv-size`,
  `--max-num-seqs`, `--prefill-step-size`). Bench config pattern:
  `configs/base.json`; launch/stop/probe: `scripts/{launch,stop}-mlx.sh`,
  `scripts/mem-macos.sh` (phys_footprint includes Metal wired memory).
- **Verification discipline** (from the skill): one change per experiment,
  controls interleaved, medians of ≥6, AC power only, cap memory (36 GiB),
  needle gates before "best", log failures in `PLAN.md`.

## 3. Verified facts about Inkling-2-Mini

All from `thinkingmachines/Inkling-2-Mini` @ `c9d0283` (config.json +
model.safetensors.index.json), the fork's source, and llama.cpp @ `9c003a9`
(fetched 2026-11-25).

**Identity:** 26.4B params, 3.4B active, **5.4B effective active** (loop adds
~1B per extra pass × 2). Dense-MoE text + ViT vision, native multimodal, 1M
context, reasoning model (thinking on by default;
`mlx_vlm/tools/parsers/inkling.py` exists; fork main streams Inkling
tool-calls). Tokenizer: 200K-class vocab (`vocab_size 201024`,
`eos_token_id 200006`).

**Checkpoint:** 19.526 GB, single safetensors shard, 1244 tensors, modelopt
packing (config `quant_algo NVFP4`; per-expert `w13_weight_scale` /
`w2_weight_scale` / `*_outer_scale_denominator`). `model.llm.embed.weight`
(+ `embed_norm`) and `unembed.weight` separate (0.46B params each).

**Text architecture (text_config):** 24 physical layers, hidden 2304, 24 Q /
8 KV heads, head_dim 128. Layers 0–5 pre, **6–17 loop core (12 layers,
shared weights, 3 passes)**, 18–23 post → 48 layer applications. MoE per
layer: 128 routed experts, top-8, 0 shared, intermediate 1152, fused stacked
expert tensors, `route_scale 8.0`. Attention: hybrid local/global (20 local
layers, sliding window 1024, sconv kernel 4; non-local {5, 11, 17, 23}),
relative-position attention (d_rel 16, rel_extent 1024), log scaling, L2
clips, attention sinks, output gates, attn-res blocks (1 per 6 layers).
Softcaps: logits 30.0, mlp gate 4.0, up 25.0. Loop config:
`core_start 6, core_end 18, num_passes 3, gate_rank 64, is_varloop false`,
with 12 controller tensors per core layer
(`controllers.{attn,k,v,mlp}.{down,gain,forget}_weight`).

**MLX support (verified in the fork and in PyPI 0.8.1):**
- Routing: `mlx_vlm/utils.py:82: "inkling2_loop": "inkling2"`.
- `models/inkling2/config.py` parses and validates `LoopConfig`
  (core_start/core_end alignment with attn-res blocks, num_passes ≥ 1,
  gate_rank ≥ 1, fixed loop only; raises for Mini without loop_config).
- `models/inkling2/language.py` implements the loop: controllers
  (≈641–690), per-core-layer `message_queries` (≈710–713), bank
  construction/mixing with softmax attention over the bank (≈742–761),
  loop-aware speculative verify (≈770), loop-aware cache behavior (≈948).
- `models/inkling2/inkling2.py` handles modelopt scales
  (`outer_scale_denominator`, line 199) and
  `middle_cycle_cross_pass_queries` tensor remap (line 210).
- Fork `main` @ `e8035ae3` adds Mini-specific kernels: grouped-expert matvec
  (#2798), attention-head prep (#2796), sconv kernel, Inkling tool-call
  streaming (#2810). **These are absent from PyPI 0.8.1.**
- Speculative scaffolding exists: `speculative/drafters/{inkling_mtp,
  inkling2_dspark}/`.
- Loop-cache layout (from the llama.cpp reference, which the MLX code
  mirrors): 48 slots = 24 + (3−1)×12, shared weights, per-slot KV + conv
  state.

**MTP:** config present, **no weights in the checkpoint** (verified).

**Cross-engine artifacts:** llama.cpp implements the arch (vendored in
`reference/llama-cpp/`); GGUF: `DevQuasar/thinkingmachines.Inkling-2-Mini-GGUF`
@ `fd1ba3a` (Q4_K_M 17.48 GB + mmproj 0.64 GB; Q2_K–Q8_0).

**Quality:** Terminal-Bench 2.1 = 71.6 (rank 28; Qwen3.8-27B = 73.0, rank 26).
Treat as pending our own gates.

## 4. Physics (ceiling) — what "done" can expect

Decode is memory-bound. Effective bytes/token at 4-bit-class density
(≈0.5625 B/param) with the loop re-reading the 12 core layers 3×:
≈ 5.4B effective active × 0.5625 ≈ **3.0 GB/token** →
**c1 ceiling ≈ 440 / 3.0 ≈ 145 tok/s** (accepted by the user). KV is mostly
bounded (window 1024 / rel_extent 1024 on 20 layers); unbounded-layer cost
at long context is READ-ME #5. Prefill is compute-bound: ~2×5.4B FLOPs/token
→ practical ceiling ≈ 0.5×15.2e12 / 1.08e10 ≈ **~700 tok/s**. c4 aggregate:
weights amortize across streams — several hundred tok/s ceiling; measure,
don't promise. Write `configs/ceiling-inkling2-mini-4bit-m4max.json` from
measured per-tensor bytes (`scripts/measure_metal.py` pattern) once the
quant exists.

## 5. Build plan (v2)

**Phase 1 — Smoke test from the fork.** `uv tool install --editable
/Users/ka/github/existential-birds/mlx-vlm --force`; serve the NVFP4
checkpoint; confirm load + loop + sane outputs. Log findings in `PLAN.md`.
If something breaks: minimal fix on a fork branch (this is now our only
"implementation" risk).

**Phase 2 — Quantization choice.** Test the modelopt dequant path on one MoE
layer; run `mlx_vlm.convert --q-bits 4` (~15–16 GB) if affine-4-bit wins the
bytes/quality trade; keep NVFP4 path working. Publish as the first
mlx-community quant if it's clean (user decision).

**Phase 3 — Validation gates (the skill's).** Fixed prompt panel, greedy
agreement sanity, no-looping detector, needle retrieval at 32K and 131K,
concurrent full-KV soak under the 36 GiB cap. Optional: cross-engine
statistical A/B vs llama.cpp Q4_K_M.

**Phase 4 — Hill-climb (the main event).** `configs/inkling-2-mini-4bit.json`
(extend `scripts/launch-mlx.sh` to take a model); E0 baseline with controls;
then levers per `references/mlx.md` order: `--kv-bits/--kv-quant-scheme`
(read the llama.cpp 16/64-layer q8_0 regression precedent first),
`--prefill-step-size`, `--max-num-seqs` + c4, drafter if weights are found
(READ-ME #4). The Mini-specific kernels are the baseline, not a lever —
they're already merged.

**Phase 5 — Upstream.** PR any fork fixes to `Neywa-Labs/mlx-vlm` (per its
CONTRIBUTING.md); optionally contribute the quant to mlx-community; feed
measured levers back into the vendored skill's `references/mlx.md`.

## 6. Risks

- The upstream Mini support may be fresh and lightly tested → smoke test
  first; expect small bugs; fix in the fork (that's what it's for).
- NVFP4 dequant subtleties (outer_scale_denominator semantics) → isolated
  tensor tests before trusting conversion.
- No drafter weights → no speculation in v1 (accepted with the 145 ceiling).
- KV accounting at long context unknown until READ-ME #5 → size the 131K
  gate carefully under the 36 GiB cap.
- If the smoke test reveals the loop path is actually broken/incomplete
  upstream, fall back to the v1 plan: port against the llama.cpp oracle
  (reference files vendored) — or switch targets to the Ornith fallback.

## 7. Reference index

| Artifact | Where |
|---|---|
| Loop implementation (upstream reference) | `reference/llama-cpp/inkling2-loop.cpp` (llama.cpp @ 9c003a9, MIT) |
| HF→GGUF tensor map (reference) | `reference/llama-cpp/inkling2_conversion.py` (same) |
| Target checkpoint | `thinkingmachines/Inkling-2-Mini` @ c9d0283 (config, index) |
| Cross-engine GGUF | `DevQuasar/thinkingmachines.Inkling-2-Mini-GGUF` @ fd1ba3a (Q4_K_M + mmproj) |
| MLX implementation (working copy) | `/Users/ka/github/existential-birds/mlx-vlm` (`existential-birds/mlx-vlm`, base `main` @ e8035ae3, upstream `Neywa-Labs/mlx-vlm`) |
| Skill (discipline, harness) | `skills/inference-optimisation/` (SKILL.md, scripts/, references/) |
| Hardware measurement | `scripts/measure_metal.py` → `THEORY.md` |
| Experiment log | `PLAN.md` |

Fetch commands (pin-verify):

```bash
curl -s https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json
curl -s https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/model.safetensors.index.json
# fork Mini-related history:
git -C /Users/ka/github/existential-birds/mlx-vlm log --oneline -i --grep=inkling
```
