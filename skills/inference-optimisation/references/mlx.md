# MLX on Apple Silicon

Serving and hill-climbing MLX models on M-series Macs. Read together with
`gotchas.md` (Apple entry) and `methodology.md` (unified-memory accounting).

## Inventory

- **Device:** `system_profiler SPDisplaysDataType` (GPU cores), `sysctl hw.memsize`.
  Full-core M4 Max: 546 GB/s spec; measure, don't trust (below).
- **Wired memory limit:** `sysctl iogpu.wired_limit_mb` (0 = system default).
  This caps total GPU-addressable memory (weights + KV + buffers), not just
  "GPU memory". Raising it needs `sudo`; record the old value and restore it
  when done (ground rule 3).
- **Memory accounting:** Metal allocations live in the process's
  `phys_footprint`, so a per-PID footprint probe is honest here — unlike Linux
  cgroups. Sample every few seconds and kill at the cap.
- **Bandwidth:** device copy of a ≥4 GiB buffer, read+write, min of several
  reps (clock ramp-up and Metal buffer-cache effects discard the first runs).
- **Peak FLOPs:** fp16 matmul at 4096³ and 8192³; the larger one shows the
  thermal/clock drop.

## Serving stack

- **Qwen3.8-27B is a native VLM** (`mlx-community/Qwen3.8-27B-*` has processor
  configs): the runtime is `mlx-vlm`, not `mlx-lm`. Text-only traffic still
  loads the vision tower — it costs startup and resident memory.
- The OpenAI-compatible server is what the harness benchmarks; **it must
  report `usage` on streamed responses** or token counts silently become
  chunk counts. Verify before trusting any number; patch or wrap the server
  if it doesn't.
- **Quants (mlx-community):** affine 4-bit/8-bit group-64, mxfp4, nvfp4,
  mxfp8, OptiQ/oQ variants. Decode reads weights once per token, so
  bytes/token is the format choice: affine 4-bit ≈ 0.5625 B/param.
- **MTP draft adapters** (`Qwen3.8-27B-MTP-*`): drafter-only safetensors
  (~0.24 GB at 4-bit). The target supplies embeddings and LM head at runtime,
  so the draft adds almost no bytes and there is no separate draft head to
  prune. MTP block size 3 caps the draft depth.

## Levers [S: mlx]

- **MTP speculative decoding is nearly free by ceiling arithmetic** (draft
  ≈ 0.24 GB vs ~15 GB target): sweep draft depth 1–3 per concurrency and
  context, as everywhere else. Base M-chips lost with deep drafts
  (bandwidth-poor); Max-class chips at 546 GB/s have headroom — measure.
- **KV cache dtype:** this family has full attention in only 16 of 64 layers
  (4 KV heads × 256 head dim): fp16 KV = 64 KiB/token → ~16 GiB per 256K
  stream. Quantized KV halves that; but on a 16/64-layer hybrid, llama.cpp
  q8_0 KV regressed decode 45→27 tok/s — treat as a trade to measure, not a
  default.
- **GDN state slots cap concurrency, not KV tokens:** 48 linear-attention
  layers carry fixed-size recurrent state per stream. Size slots for N
  streams deliberately and prove cN actually runs N.
- **Wired limit for long-context pools:** a 131K–256K gate + concurrent soak
  may not fit under the default wired limit. Raise it, run the gate, restore
  it, and say so in the report.
- **Metal buffer cache:** `mx.metal.set_cache_limit` / `clear_cache` bounds
  resident buffers; keep ≥1.5–2 GB free or the machine livelocks.
- **Prompt/prefix caching:** verify with a repeated-prefix TTFT drop
  (warm ≪ cold) and by defeating it with unique prefixes when measuring
  cold prefill.
- **Kernel-launch overhead:** no CUDA-graph equivalent; each decode step pays
  Metal dispatch. Speculative batches amortize it — judge per-step ms, not
  acceptance alone.
- **First-run Metal kernel compilation** inflates TTFT and startup: warm up
  with an unmeasured request, and discard the cold boot.

## Gotchas [S: mlx]

- Launch the server **detached** (`nohup`/`setsid`, PID file) so agent
  restarts don't kill it; probe and stop by PID, never `pkill -f`.
- Stop via drain-then-timeout; SIGTERM an idle server, never one mid-prefill.
- Battery or low-power mode moves results 8–10%; the harness refuses battery.
- Pin the model repo revision (the adapter and target must come from the same
  Qwen3.8-27B checkpoint) and the `mlx-vlm` version — draft wiring is
  version-sensitive.
- Streaming may emit several tokens per chunk with MTP; count from usage.

## Measured Precedents — Looped MoE (`Inkling-2-Mini`, M4 Max 48 GB)

- **Looped-layer dispatch floor breaks naive bytes/token ceilings:** On a 48-visit
  looped MoE (`24` physical layers, middle `12` × `3` passes, `3.03 GB/tok` in
  mixed 4-bit dense + native NVFP4/MXFP8 experts), pure weight kernels took
  `9.9 ms/tok` (`70%` of `440 GB/s`), while `9,951` MLX ops across 48 visits
  added `13.7 ms/tok` (`58%` of step time) in loop controllers, `AttentionResidual`
  bank mixing, and 192 short-convolutions — capping `c1` at `46.5 tok/s` vs a
  naive `139 tok/s` bandwidth ceiling. Batching `(attn, k, v)` loop controllers
  and `@mx.compile` saved ~`1.5–2 ms/step`.
- **Sparse MoE `cN` ceiling must count expert union, not single-stream bytes:**
  At `B=4` on top-8/128 MoE, 4 independent streams touch `128 × (1 - (120/128)^4)
  = 29.0` unique experts/layer (`3.62×` single-stream expert bytes), lowering the
  true `c4` bandwidth ceiling from `836` (dense-sharing assumption) to `224 tok/s`
  (`93.3 tok/s` measured).
- **DSpark / block-diffusion break-even on looped MoE:** `L`-token verify widens
  the active expert union across all 48 visits (`57 ms/round` at `block_size=5`
  vs `21.5 ms` 1-token decode → break-even `a* = 2.48`). Shallow blocks (`5`)
  beat deep blocks (`8`: `100 ms/round`, `a* = 4.35`) by `+79%` across the panel,
  winning on code (`54.3 tok/s`, `+25%`) but losing on `T=0.6` prose (`26–34 tok/s`)
  and regressing `c4` (`-58%`).
- **Bound `--prefill-step-size` when attention materializes `[B, H, LQ, S]` masks:**
  Models with relative-position or banded float masks allocate `B × H × LQ × S × 2 B`
  per global layer (`24.3 GiB` at `LQ=2048, S=131K` incl. sink concat). Lowering
  `--prefill-step-size` to `512` + `--cache-limit 4 GiB` cut peak 32K footprint
  `30.1 → 23.9 GiB`, improved 4K/32K prefill (`590 / 547 tok/s`), and fit 131K
  (`30.1 GiB`) and `2 × 67K` concurrent soak (`25.2 GiB`) under 36 GiB without
  raising `iogpu.wired_limit_mb`.
- **Clear spent `left_padding` on recurrent/conv batch caches:** `ArraysCache`
  retained negative `left_padding` arrays after prefill on unequal-length batches,
  disabling fused `ShortConvolution` Metal kernels and adding `+7,368` ops/step
  until cleared once `advance >= max(padding)`.
- **macOS probes & readiness:** `footprint -p <pid>` suspends a 20 GB Metal task
  for up to `29 s`; query `libproc.proc_pid_rusage` (`RUSAGE_INFO_V4.ri_phys_footprint`,
  `< 1 ms`, zero suspension) instead. Poll `mlx_vlm.server`'s `/health`
  (`model_ready: true`) rather than `/v1/models`, which synchronously runs
  `huggingface_hub.scan_cache_dir()` on every call.
