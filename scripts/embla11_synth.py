#!/usr/bin/env python3
"""A synthetic Embla 1.1 model directory made from Embla 1 (Embla1-4bit): Embla 1.1's shapes and config.json, for
the engine's 1.1 paths before trained weights exist. No dependencies beyond the standard library.

    python3 scripts/embla11_synth.py <Embla 1 dir> <out dir> [--preserve]

On top of Embla 1's weights (docs/EMBLA_1_1.md):
  * router mix on MoE layers 0-15: `router_res_query` = the layer's mlp_res_query (the same routing);
  * layer 0 MoE: its dense MLP (2048) as its shared expert (`mlp.shared_fc` / `shared_proj`), plus 96 x 512 routed
    experts top-8 and a router: layer 1's (--preserve: the routed down projections zero, so the layer computes the
    dense MLP);
  * K/V mix at layer 16: `transformer.h.16.kv_res_query` = its attn_res_query; without --preserve layers 24 / 28 take
    their K/V from layer 16 (their own c_k / c_v, now on layer 16's mix); with --preserve they stay on layer 7;
  * the loop over [8, 16) x 2: `loop_{attn,mlp,router}_res_query` = the first pass's queries, `transformer.loop_alpha`
    [d] f32 (0 with --preserve, else 0.1), `transformer.loop_src_bias` [4] f32 = -log 2.

--preserve computes Embla 1's function (up to rounding) with every 1.1 path on. The engram table and tokenizer are
symlinked, model.safetensors is written in full (a new file, Embla 1's tensors copied).
"""
import argparse
import json
import math
import os
import shutil
import struct
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--preserve", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = open(src / "model.safetensors", "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None)
    base = 8 + n

    def raw(k: str) -> bytes:
        s, e = hdr[k]["data_offsets"]
        f.seek(base + s)
        return f.read(e - s)

    # name -> (dtype, shape, source: base tensor name or bytes)
    tensors = {k: (v["dtype"], v["shape"], k) for k, v in hdr.items()}
    d = 2048

    def alias(new: str, old: str, zero: bool = False):
        dt, sh, _ = tensors[old]
        tensors[new] = (dt, sh, bytes(len(raw(old))) if zero else old)

    for l in range(16):
        alias(f"transformer.h.{l}.router_res_query", f"transformer.h.{l}.mlp_res_query")
    alias("transformer.h.16.kv_res_query", "transformer.h.16.attn_res_query")
    for l in range(8, 16):
        alias(f"transformer.h.{l}.loop_attn_res_query", f"transformer.h.{l}.attn_res_query")
        alias(f"transformer.h.{l}.loop_mlp_res_query", f"transformer.h.{l}.mlp_res_query")
        alias(f"transformer.h.{l}.loop_router_res_query", f"transformer.h.{l}.mlp_res_query")
    p0, p1 = "transformer.h.0.mlp.", "transformer.h.1.mlp."
    for part in ("weight", "scales", "biases"):
        alias(p0 + "shared_fc." + part, p0 + "c_fc." + part)
        alias(p0 + "shared_proj." + part, p0 + "c_proj." + part)
        alias(p0 + "experts_fc." + part, p1 + "experts_fc." + part)
        alias(p0 + "experts_proj." + part, p1 + "experts_proj." + part, zero=a.preserve)
        del tensors[p0 + "c_fc." + part], tensors[p0 + "c_proj." + part]
    alias(p0 + "router.weight", p1 + "router.weight")
    alias(p0 + "expert_bias", p1 + "expert_bias")
    tensors["transformer.loop_alpha"] = ("F32", [d], struct.pack(f"<{d}f", *([0.0 if a.preserve else 0.1] * d)))
    tensors["transformer.loop_src_bias"] = ("F32", [4], struct.pack("<4f", *([-math.log(2.0)] * 4)))

    names = sorted(tensors)
    size = lambda k: len(tensors[k][2]) if isinstance(tensors[k][2], bytes) else hdr[tensors[k][2]]["data_offsets"][1] - hdr[tensors[k][2]]["data_offsets"][0]
    h, off = {}, 0
    for k in names:
        h[k] = {"dtype": tensors[k][0], "shape": tensors[k][1], "data_offsets": [off, off + size(k)]}
        off += size(k)
    if meta:
        h["__metadata__"] = meta
    hb = json.dumps(h, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out / "model.safetensors", "wb") as o:
        o.write(struct.pack("<Q", len(hb)))
        o.write(hb)
        for k in names:
            s = tensors[k][2]
            o.write(s if isinstance(s, bytes) else raw(s))

    cfg = json.loads((src / "config.json").read_text())
    cfg["mlp_layer_types"][0] = "sparse"
    cfg["shared_expert_layer_sizes"] = {"0": 2048}
    cfg["moe_router_mix"], cfg["moe_router_mix_layers"] = True, 16
    cfg["kv_mix_layers"] = [16]
    if not a.preserve:
        cfg["kv_sources"] = {"7": 7, "16": 16, "20": 16, "24": 16, "28": 16}
    cfg["loop_layers"], cfg["loop_passes"], cfg["loop_pass_form"] = [8, 16], 2, "full"
    for q in ("quantization", "quantization_config"):
        if q in cfg:
            cfg[q]["transformer.h.0.mlp.router"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("engram_table.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json"):
        if (src / name).exists() and not (out / name).exists():
            os.symlink(os.path.realpath(src / name), out / name)
    print(f"{out}: {len(names)} tensors, {off / 1e9:.2f} GB" + (" (preserve)" if a.preserve else ""))


if __name__ == "__main__":
    main()
