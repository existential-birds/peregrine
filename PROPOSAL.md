# PROPOSAL — Bring Inkling-2-Mini to MLX: implement `inkling2_loop`

Status: proposed (2026-11-25). Owner: this repo. Companion log: `PLAN.md`.

---

## READ ME FIRST — instructions for the next agent

You are implementing (or researching before implementing) MLX support for
Thinking Machines' **Inkling-2-Mini**, a looped-layer MoE VLM. This proposal
contains everything verified so far. **Everything in §3 is verified with the
commands given; everything in this section is open.** Resolve these in order
before writing model code — they are ranked by how much they gate the design.

1. **Loop-pass dataflow.** The exact per-pass computation is in
   `reference/llama-cpp/inkling2-loop.cpp` (123 lines, vendored — read it
   fully). Known so far: every pass starts from the *prelude* output (layers
   0–5); each pass's input is `mix(bank, bank_out, msg_query)` where `bank` is
   the previous pass's core-block outputs after the first core block, plus its
   output; per-layer controllers (`{attn,k,v,mlp}.{down,gain,forget}_weight`)
   modulate Q/K/V/FFN. You need the exact `mix()` and controller math —
   follow `build_layer` / `mix` into `llama-model-inkling2.cpp` and the graph
   builder in the llama.cpp repo at the pinned commit.
2. **`msg_query` weight source.** llama.cpp loads a per-core-layer
   `LOOP_MSG_QUERY` tensor `{n_embd, 4}`. The Mini checkpoint has
   `model.llm.attn_res.attn_queries.{0..23}` and the 12 controller tensors
   per core layer, but no tensor literally named msg_query. The mapping is in
   `reference/llama-cpp/inkling2_conversion.py` (559 lines, vendored — the
   HF→GGUF name map). Confirm whether attn_res queries double as message
   queries.
3. **Cache-slot layout.** 48 logical slots: `n_slot = 24 + (3−1)×12`
   (physical 24 + two extra copies of the 12 core layers). Slot→physical
   mapping is `layer_phys(il)` (first lines of the vendored .cpp). Each slot
   keeps **its own KV and conv state**; local slots are sliding-window (1024).
   Decide how to express this in MLX: extend `mlx_vlm`'s cache abstraction or
   keep a model-internal 48-slot list.
4. **Attention block internals.** Mini uses exotic attention: relative-position
   attention (`d_rel 16`, `rel_extent 1024`, `log_scaling_n_floor/alpha`),
   L2 clips on q/rel, attention sinks (`attn_sink_proj`), output gates
   (`gate_attn`), short-conv K/V (`k_sconv`/`v_sconv`, kernel 4), attn-res
   blocks (`n_layers_per_attn_res_block 6`, `attn_res.attn_queries`,
   `attn_res_normalize_values`), softcaps (logit 30.0, mlp gate 4.0, up 25.0).
   MLX's `inkling2` (non-loop sibling) very likely implements most of this —
   **diff Mini's `text_config` against Inkling-2-Small's config.json** and read
   `…/mlx_vlm/models/inkling2/language.py` to find the deltas.
5. **MTP drafter weights.** `mtp_config` (block_size 8, 3 draft layers, Markov
   `rnn_hidden` head, `target_layer_ids [5,23,41,48]` — note 41,48 are
   loop-expanded slots) is in the config, but **zero mtp tensors exist in the
   checkpoint** (verified). Search thinkingmachines' org for a Mini drafter
   repo; if none exists, ship v1 without speculation and revisit
   (`mlx_vlm/speculative/drafters/inkling_mtp/` already exists as scaffolding).
6. **Conversion/quantization path.** The checkpoint is modelopt-style packed
   (fused `w13`/`w2` experts + `*_weight_scale` + `*_outer_scale_denominator`;
   `quantization_config: NVFP4`). MLX's inkling2 already divides by
   `outer_scale_denominator` (inkling2.py:199). Verify `mlx_vlm.convert`
   dequantizes these tensors correctly (test one layer in isolation) before
   requantizing to affine 4-bit.
7. **Which layers have unbounded KV.** local_layer_ids covers 20/24 layers
   (sliding window 1024 + rel_extent 1024 → bounded). Layers 5, 11, 17, 23 are
   the non-local ones. Determine their KV growth (this sets the real KV
   ceiling at 131K/1M context, and whether `--kv-bits` matters here).
8. **llama.cpp build for the oracle.** Homebrew's llama.cpp may lag the
   pinned commit. Verify `llama-server --version` knows `inkling2-loop`, else
   build from source at the pinned SHA (cmake build on this Mac is quick).

Also read: `skills/inference-optimisation/SKILL.md` (the discipline this repo
follows), `PLAN.md` (experiment log + model-selection context), `THEORY.md`
(measured hardware), `references/mlx.md` (MLX-specific levers/gotchas).

---

## 1. Decision and rationale

The goal was 1000+ tok/s. Ceilings-first analysis (see `PLAN.md` "Model
selection") showed no lab ship hits that single-stream on this machine. The
user accepted the trade: **Inkling-2-Mini at ~145 tok/s ceiling for the
novelty of the architecture** — a looped-layer MoE reasoning VLM, near-Qwen3.8-27B
quality (Terminal-Bench 2.1: 71.6 vs 73.0) at ~1/5 the bytes-per-token.

This proposal covers building what is missing:
- **(1)** an `inkling2_loop` model implementation for MLX (`mlx-vlm`), and
- **(2)** a fitting quantized MLX checkpoint (the official repo is 19.5 GB
  modelopt NVFP4 — it fits as-is, but we also want an affine 4-bit ~15–16 GB
  for the speed play and MLX-native format).

Fallback/comparator if this stalls: Ornith-1.5-35B-A3B MLX-4bit + DFlash
(206–508 tok/s ceiling, first-party MLX, dflash draft) — analysis in `PLAN.md`.

## 2. Project context (all a future model needs)

- **Repo:** `/Users/ka/github/existential-birds/mlx-experiment` (git, `main`).
  Working dir of the vendored `inference-optimisation` skill
  (`skills/inference-optimisation/`); harness `scripts/bench.py` is
  engine-agnostic (any OpenAI-compatible server), `ceiling.py` computes
  theoretical ceilings, `queue.sh` serializes runs. Experiment log: `PLAN.md`.
- **Machine:** MacBook Pro Mac16,6, Apple M4 Max (16-core CPU, 40-core GPU),
  48 GB unified, macOS 26.6.1. Measured (2026-11-25, `scripts/measure_metal.py`):
  device copy **438–454 GB/s** (theory uses 440), fp16 GEMM **15.2 TFLOPS**
  (no thermal drop at 8192³). Wired limit at system default
  (`iogpu.wired_limit_mb=0`, untouched).
- **MLX stack:** `mlx 0.32.4` (pyenv python3.12 for measurement), `mlx-vlm`
  via `uv tool` → venv at `/Users/ka/.local/share/uv/tools/mlx-vlm`
  (python3.14). Server: `mlx_vlm.server` (OpenAI-compatible, port 8080;
  flags include `--draft-model --draft-kind {dflash,eagle3,mtp}
  --draft-block-size`, `--kv-bits --kv-quant-scheme {uniform,turboquant}`,
  `--max-kv-size`, `--max-num-seqs`, `--prefill-step-size`). Bench config
  pattern: `configs/base.json`; launch/stop/probe: `scripts/{launch,stop}-mlx.sh`,
  `scripts/mem-macos.sh` (phys_footprint includes Metal wired memory).
- **Verification discipline** (from the skill): one change per experiment,
  controls interleaved, medians of ≥6, AC power only, cap memory (36 GiB),
  needle gates before "best", log failures in `PLAN.md`.

## 3. Verified facts about Inkling-2-Mini

All from `thinkingmachines/Inkling-2-Mini` @ `c9d0283` (config.json +
model.safetensors.index.json, fetched 2026-11-25) and llama.cpp @ `9c003a9`.

**Identity:** 26.4B params, 3.4B active, **5.4B effective active** (loop adds
~1B per extra pass × 2). Dense-MoE text model + ViT vision, native multimodal.
1M-token context (`model_max_length 1048576`). Reasoning model (thinking on by
default; `mlx_vlm/tools/parsers/inkling.py` exists). Tokenizer: 200K-class
vocab (`vocab_size 201024`, `eos_token_id 200006`, llama-3 family).

**Checkpoint:** 19.526 GB, single safetensors shard, 1244 tensors, bf16
metadata with modelopt packing (config says `quant_algo NVFP4` — scales
present per expert tensor). embed (`model.llm.embed.weight`, plus
`embed_norm`) and `unembed.weight` are separate (0.46B params each).

**Text architecture (text_config):**
- 24 physical layers, hidden 2304, 24 Q heads / 8 KV heads / head_dim 128.
- Structure: layers 0–5 pre, **6–17 loop core (12 layers, shared weights,
  3 passes)**, 18–23 post. Effective 48 layer applications.
- MoE per layer: 128 routed experts, top-8, **0 shared experts**,
  intermediate 1152 (tiny experts), `route_scale 8.0`, `use_gate_bias`,
  fused stacked expert tensors (`mlp.experts.w13_weight`, `w2_weight` +
  modelopt scales).
- Attention: hybrid local/global — `local_layer_ids` = 20 layers
  (sliding_window 1024, sconv kernel 4); non-local = {5, 11, 17, 23};
  relative-position attention (d_rel 16, rel_extent 1024), log scaling
  (n_floor 8192, alpha 0.1), q/rel L2 clips, attn sinks, output gates,
  attn-res blocks (1 per 6 layers, `attn_res.attn_queries.{i}`).
- Softcapping: final logits 30.0, mlp gate 4.0, mlp up 25.0.
- **Loop config:** `core_start 6, core_end 18, num_passes 3, gate_rank 64,
  is_varloop false`. Per-core-layer controller tensors (12):
  `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight`.

**Loop semantics (from vendored llama.cpp `inkling2-loop.cpp`):**
- 48 cache slots: `n_slot = n_phys + (n_passes−1)×(core_end−core_start)`
  = 24 + 2×12. "Share the physical weights, each cache slot keeps its own
  KV and conv state."
- The loop graph **subclasses the plain inkling2 graph** (`llama_model_inkling2_loop::graph
  : llama_model_inkling2::graph`) — the same inheritance strategy is the
  recommended MLX design.
- "Every pass starts from the prelude output"; pass p>0 input mixes the
  previous pass's bank (its core blocks after the first + its output) via
  `mix(bank, bank_out, loop.msg_query)`; per-layer, `build_layer(il, ip, cur,
  blocks, mix(...), &loop)` applies the controller modulation.
- The loop file is only **123 lines** — the loop machinery itself is small;
  everything else reuses inkling2.

**Vision (vision_config):** `model_type "inkling2_vision"` — same family as
Inkling-2-Small → `mlx_vlm/models/inkling2/vision.py` should be reusable.
23-layer ViT, hidden 1024, patch 20, projector hidden 4096.

**MTP:** config present (3 layers, block_size 8, Markov rnn_hidden head,
targets [5, 23, 41, 48]) but **no weights in the checkpoint**.

**Oracle artifacts (for validation):** llama.cpp implements the arch
(`src/models/inkling2-loop.cpp`, `conversion/inkling2.py`, both vendored in
`reference/llama-cpp/`). GGUF: `DevQuasar/thinkingmachines.Inkling-2-Mini-GGUF`
@ `fd1ba3a` — Q4_K_M 17.48 GB (+ 0.64 GB mmproj), Q2_K…Q8_0 available.

**Quality:** Terminal-Bench 2.1 = 71.6 (rank 28; Qwen3.8-27B = 73.0, rank 26).
Terminal-bench not the only axis: TML card claims strong reasoning/effort
scaling. Treat quality as pending our own gates.

## 4. Physics (ceiling) — what "done" can expect

Decode is memory-bound. Effective bytes/token at 4-bit-class density
(≈0.5625 B/param) with the loop re-reading the 12 core layers 3×:
≈ 5.4B effective active × 0.5625 ≈ **3.0 GB/token** →
**c1 ceiling ≈ 440 / 3.0 ≈ 145 tok/s** at short context (accepted by the
user). KV is mostly bounded (window 1024 / rel_extent 1024 on 20 layers);
the unbounded-layer cost at long context is open question §READ-ME-#7.
Prefill is compute-bound: ~2×5.4B FLOPs/token → practical ceiling
0.5×15.2e12 / 1.08e10 ≈ **~700 tok/s**. c4 aggregate: weights amortize
across streams — ceiling several hundred tok/s; measure, don't promise.

A `configs/ceiling-inkling2-mini-4bit-m4max.json` should be written with
measured per-tensor bytes (safetensors headers, `scripts/measure_metal.py`
pattern) once the quant exists.

## 5. Build plan

**Phase 0 — Oracle (do first; needed regardless).**
Install/build llama.cpp @ `9c003a9` with inkling2-loop; download the Q4_K_M
GGUF + mmproj; serve on port 8081; run `scripts/bench.py` against it
(config: `configs/inkling-llamacpp-q4km.json`, `base_url
http://127.0.0.1:8081`). This gives (a) a baseline, (b) the oracle outputs
for Phase 3, (c) a same-day engine comparison for methodology.md.

**Phase 1 — Research.** Resolve READ-ME items 1–8. Deliverable: a short
`RESEARCH.md` in this repo with the loop dataflow written out (pass inputs,
controller math, cache slot map, tensor name table HF→GGUF→MLX).

**Phase 2 — `mlx_vlm/models/inkling2_loop/`.** New package:
`config.py` (parse nested `text_config` + `loop_config`), `language.py`
(subclass/reuse inkling2 layer primitives; loop execution over 3 passes;
48-slot cache with per-slot KV + conv state; controller modulation),
`__init__.py` + registry wiring (`models/__init__.py`,
`tools/registry.py`). Reuse `inkling2` vision/processing/parser. Target
diff size: a few hundred lines, mirroring the 123-line llama.cpp delta.

**Phase 3 — Equivalence validation.** Statistical, not bit-exact (different
quant formats): greedy top-1 token agreement ≥ 99% over the fixed prompt
panel (100+ prompts, thinking off), logit KL at divergence points (the
skill's methodology: a 0.016-nat top-2 tie is not a bug), plus output sanity
(looping detector, needle retrieval at 32K/131K). Any mismatch: compare
layer-by-layer against llama.cpp debug dumps before touching kernels.

**Phase 4 — Quantize.** `mlx_vlm.convert -hf thinkingmachines/Inkling-2-Mini
--q-bits 4` → affine 4-bit (~15–16 GB). Sanity first on one MoE layer
(READ-ME #6). Keep the original NVFP4 load path working too — A/B them
(bytes/token vs quality gates).

**Phase 5 — Integrate & hill-climb.** New `configs/inkling-2-mini-4bit.json`
(like `configs/base.json`), extend `scripts/launch-mlx.sh` to take the model
as a parameter; ceiling config from measured tensor bytes; then the standard
loop: E0 baseline (controls ×2), then levers in `references/mlx.md` order —
KV-bits/scheme, `--prefill-step-size`, `--max-num-seqs`/c4, draft (if Phase-1
finds drafter weights), all gated by needle tests at 131K. Memory cap 36 GiB.

**Phase 6 — Upstream.** PR the `inkling2_loop` model to mlx-vlm upstream
(evidence attached: equivalence numbers, bench table). Also consider
contributing the MLX quants to mlx-community, and the hill-climb results
(levers with evidence) back to the inference-optimisation skill — its
contributing section asks for exactly this.

## 6. Risks

- Loop-KV semantics bugs → silent quality loss. Mitigation: Phase 3 gates,
  per-layer debug comparison, needle tests.
- The exotic attention (rel/log-scaling/sconv/attn-res) may differ between
  Small and Mini more than assumed → diff configs first (READ-ME #4).
- NVFP4 dequant subtleties (outer_scale_denominator semantics) → isolated
  tensor tests before conversion.
- No drafter weights → no speculation in v1 (accept; ceiling already accepted).
- Effort creep: if Phase 1 finds the delta is not the ~123-line inheritance
  the llama.cpp structure suggests, stop and report before sinking days —
  the Ornith fallback is one `configs/` file away.

## 7. Reference index

| Artifact | Where |
|---|---|
| Loop implementation (reference) | `reference/llama-cpp/inkling2-loop.cpp` (llama.cpp @ 9c003a9, MIT) |
| HF→GGUF tensor map (reference) | `reference/llama-cpp/inkling2_conversion.py` (same) |
| Target checkpoint | `thinkingmachines/Inkling-2-Mini` @ c9d0283 (config, index) |
| Oracle GGUF | `DevQuasar/thinkingmachines.Inkling-2-Mini-GGUF` @ fd1ba3a (Q4_K_M + mmproj) |
| MLX sibling implementation | `~/.local/share/uv/tools/mlx-vlm/.../mlx_vlm/models/inkling2/` |
| MLX inkling drafters | `…/mlx_vlm/speculative/drafters/{inkling_mtp,inkling2_dspark}/` |
| Reasoning parser | `…/mlx_vlm/tools/parsers/inkling.py` |
| Skill (discipline, harness) | `skills/inference-optimisation/` (SKILL.md, scripts/, references/) |
| Hardware measurement | `scripts/measure_metal.py` → `THEORY.md` |
| Experiment log | `PLAN.md` |

Fetch commands (pin-verify):

```bash
curl -s https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json
curl -s https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/model.safetensors.index.json
# llama.cpp source of the two vendored files:
https://github.com/ggml-org/llama.cpp/blob/9c003a97ab34a69cc49243f9b02014038205ffae/src/models/inkling2-loop.cpp
https://github.com/ggml-org/llama.cpp/blob/9c003a97ab34a69cc49243f9b02014038205ffae/conversion/inkling2.py
```
