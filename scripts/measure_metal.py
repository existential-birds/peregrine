#!/usr/bin/env python3
"""Step-0 hardware (+ optional checkpoint header) measurement for MLX hill-climbs.

Measures (per references/ceilings.md — measure, don't trust spec sheets):
  - device copy bandwidth (read+write over >=3 GiB, min of warm reps)
  - fp16 GEMM peak at 4096^3 and 8192^3 (the larger shows thermal/clock drop)
  - optional per-tensor byte breakdown of Hugging Face target / draft checkpoints
    via safetensors HTTP range requests (no full download)

Examples:
  python3 scripts/measure_metal.py
  python3 scripts/measure_metal.py --repo existentialbirds/Inkling-2-Mini-4bit \\
      --draft existentialbirds/Inkling-2-Mini-MTP-4bit
"""
import argparse
import json
import struct
import time
import urllib.error
import urllib.request


def device_copy_gbs(mx, nbytes=3 << 30, reps=7):
    a = mx.zeros((nbytes // 2,), dtype=mx.float16)
    mx.eval(a)
    times = []
    for i in range(reps):
        b = a + 1
        t0 = time.perf_counter()
        mx.eval(b)
        times.append(time.perf_counter() - t0)
        print(f"  copy rep {i}: {2 * nbytes / times[-1] / 1e9:7.0f} GB/s")
    return 2 * nbytes / min(times[1:]) / 1e9  # discard cold rep 0


def gemm_tflops(mx, n, reps=5):
    a = mx.random.normal((n, n), dtype=mx.float16)
    b = mx.random.normal((n, n), dtype=mx.float16)
    mx.eval(a, b)
    times = []
    for i in range(reps):
        t0 = time.perf_counter()
        mx.eval(a @ b)
        times.append(time.perf_counter() - t0)
        print(f"  gemm {n}^3 rep {i}: {2 * n**3 / times[-1] / 1e12:6.1f} TFLOPS")
    return 2 * n**3 / min(times[1:]) / 1e12  # discard cold rep 0


def discover_shards(repo, revision="main"):
    index_url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors.index.json"
    try:
        with urllib.request.urlopen(index_url, timeout=60) as r:
            weight_map = json.loads(r.read()).get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        if shards:
            return shards
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return ["model.safetensors"]


def fetch_header(repo, fname, revision="main"):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{fname}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=60).read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read(n))


def tensor_bytes(repo, revision="main"):
    files = discover_shards(repo, revision=revision)
    groups = {"lm_head": 0, "embed": 0, "model": 0}
    total = 0
    for f in files:
        for name, meta in fetch_header(repo, f, revision=revision).items():
            if "data_offsets" not in meta:
                continue
            size = meta["data_offsets"][1] - meta["data_offsets"][0]
            total += size
            if "lm_head" in name or "unembed" in name:
                g = "lm_head"
            elif "embed" in name:
                g = "embed"
            else:
                g = "model"
            groups[g] += size
    return groups, total


def main():
    p = argparse.ArgumentParser(description="Measure Apple Silicon Metal bandwidth/FLOPs and optional HF checkpoint sizes.")
    p.add_argument("--repo", help="Optional Hugging Face target model repo ID for safetensors header sizing")
    p.add_argument("--draft", help="Optional Hugging Face draft model repo ID for safetensors header sizing")
    p.add_argument("--revision", default="main", help="Hugging Face revision for --repo/--draft (default: main)")
    args = p.parse_args()

    import mlx.core as mx

    print("== device copy (read+write) ==")
    bw = device_copy_gbs(mx)
    print("== fp16 GEMM ==")
    tf4 = gemm_tflops(mx, 4096)
    tf8 = gemm_tflops(mx, 8192)

    summary = {
        "device_copy_gbs": round(bw, 1),
        "gemm4096_tflops": round(tf4, 1),
        "gemm8192_tflops": round(tf8, 1),
    }

    if args.repo or args.draft:
        print("== checkpoint sizes (safetensors headers) ==")
    if args.repo:
        groups, total = tensor_bytes(args.repo, revision=args.revision)
        print(f"  target {args.repo}: total {total/1e9:.3f} GB  "
              f"model {groups['model']/1e9:.3f}  embed {groups['embed']/1e9:.3f}  "
              f"lm_head {groups['lm_head']/1e9:.3f}")
        summary.update({
            "target_total_gb": round(total / 1e9, 3),
            "target_model_gb": round(groups["model"] / 1e9, 3),
            "target_embed_gb": round(groups["embed"] / 1e9, 3),
            "target_lm_head_gb": round(groups["lm_head"] / 1e9, 3),
        })
    if args.draft:
        dgroups, dtotal = tensor_bytes(args.draft, revision=args.revision)
        print(f"  draft {args.draft}: total {dtotal/1e9:.3f} GB  {dgroups}")
        summary["draft_total_gb"] = round(dtotal / 1e9, 3)

    try:
        info = mx.metal.get_device_info()
        print("== metal device info ==", json.dumps(info, default=str))
    except Exception as e:
        print("metal device info unavailable:", e)
    print("== summary ==")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
