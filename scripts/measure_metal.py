#!/usr/bin/env python3
"""Step-0 hardware + checkpoint measurement for the MLX hill-climb.

Measures (per references/ceilings.md — measure, don't trust spec sheets):
  - device copy bandwidth (read+write over >=4 GiB, min of reps)
  - fp16 GEMM peak at 4096^3 and 8192^3 (the larger shows thermal/clock drop)
  - per-tensor byte breakdown of the target and MTP-draft checkpoints,
    fetched from the safetensors headers (range requests, no full download)

Prints a JSON blob at the end for THEORY.md / the ceiling config.
"""
import json
import struct
import time
import urllib.request

import mlx.core as mx

REPO = "mlx-community/Qwen3.8-27B-4bit"
DRAFT = "mlx-community/Qwen3.8-27B-MTP-4bit"
SHARDS = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
          "model-00003-of-00003.safetensors"]


def device_copy_gbs(nbytes=3 << 30, reps=7):
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


def gemm_tflops(n, reps=5):
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


def fetch_header(repo, fname):
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=60).read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read(n))


def tensor_bytes(repo, files):
    groups = {"lm_head": 0, "embed": 0, "model": 0}
    total = 0
    for f in files:
        for name, meta in fetch_header(repo, f).items():
            if "data_offsets" not in meta:
                continue
            size = meta["data_offsets"][1] - meta["data_offsets"][0]
            total += size
            g = "lm_head" if "lm_head" in name else "embed" if "embed" in name else "model"
            groups[g] += size
    return groups, total


def main():
    print("== device copy (read+write) ==")
    bw = device_copy_gbs()
    print("== fp16 GEMM ==")
    tf4 = gemm_tflops(4096)
    tf8 = gemm_tflops(8192)
    print("== checkpoint sizes (safetensors headers) ==")
    groups, total = tensor_bytes(REPO, SHARDS)
    print(f"  target {REPO}: total {total/1e9:.3f} GB  "
          f"model {groups['model']/1e9:.3f}  embed {groups['embed']/1e9:.3f}  "
          f"lm_head {groups['lm_head']/1e9:.3f}")
    dgroups, dtotal = tensor_bytes(DRAFT, ["model.safetensors"])
    print(f"  draft {DRAFT}: total {dtotal/1e9:.3f} GB  {dgroups}")
    try:
        info = mx.metal.get_device_info()
        print("== metal device info ==", json.dumps(info, default=str))
    except Exception as e:
        print("metal device info unavailable:", e)
    print("== summary ==")
    print(json.dumps({"device_copy_gbs": round(bw, 1), "gemm4096_tflops": round(tf4, 1),
                      "gemm8192_tflops": round(tf8, 1),
                      "target_total_gb": round(total / 1e9, 3),
                      "target_model_gb": round(groups["model"] / 1e9, 3),
                      "target_embed_gb": round(groups["embed"] / 1e9, 3),
                      "target_lm_head_gb": round(groups["lm_head"] / 1e9, 3),
                      "draft_total_gb": round(dtotal / 1e9, 3)}))


if __name__ == "__main__":
    main()
