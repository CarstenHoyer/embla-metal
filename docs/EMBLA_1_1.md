# Embla 1.1 in embla-metal: design

Status: implemented, decode and prefill (branch `engind/embla-1.1`, 2026-11-16). The model is heimr-edge's Embla 1.1
(sevren-ai/sparsify `agent_plans/EMBLA_1_1.md`; the PyTorch truth is branch `kweind/embla-1.1-model`, re-read at
bdb8333, `_run_loop`). A reference port exists in the MLX app harness (sevren-ai/heimr-ios `engind/embla-1.1-runtime`
01c0d43). It checks against the trainer and is a map of what changes, not code to copy. Every change here is driven by
`config.json` (`Arch`), and Embla 1 must stay bit-identical at the same speed (its pipelines keep their macros; 1.1
adds new instantiations).

## As implemented

- `Arch` (Arch.swift) reads kv_sources, kv_mix_layers, moe_router_mix(_layers), mlp_layer_types,
  shared_expert_layer_sizes and loop_layers; anything else the kernels do not take is refused at load.
- b* has no kernel of its own: layer 16's prep (decode) and attention mix (prefill) compute it (`STAR`), as APPEND
  computes a completed block, and write it to slots 6 and 7. The fp32 sums: `KEEP` rows 0 (B2, pass 1), 1 (B3, pass 1,
  at pass 2's start), 2 (B2', pass 2); B3' is x itself.
- `OFFS` is needed only in prep and mix: the fused-mix GEMVs, the router and the reduce take prep's lse.
- mix `LOWREG`: the final mix over 11 sources (and prefill's layer 16 mix, 8 sources x 5 queries) needs more
  registers than 1024 threads have (832 and 576 on the M1); the same sums, reduced per source, with fewer.
- Dispatches a decode step (measured): Embla 1 209, Embla 1.1 260 (+51: pass 2's 8 layers, layer 0's MoE).
- Not in the design: on GPUs before Apple9 (M1, M2) the cross-threadgroup reads after `arrive_last` (router logits,
  reduce sums, attention partials) were stale, so Embla 1 itself decoded non-deterministically there. They go through
  atomics on those GPUs (`XTG_ATOMIC`, common.metal); Apple9 and later compile as before.
- `scripts/embla11_synth.py` makes a synthetic 1.1 from Embla1-4bit (`--preserve`: the function-preserving form).
  The preserve form matches Embla 1 up to rounding, not bit for bit: layer 0's shared expert sums its down
  projection as the MoE does, and each looped block's softmax weight is split between b and b* by the -log 2 offsets.
- `Apps/EmblaBench`: the smallest iPhone host for `embla-bench` (xcodegen).

## What Embla 1.1 changes, against this engine's plans (`Engine.buildPlans`)

1. **Router mix, MoE layers 0-15** (`moe_router_mix`, `moe_router_mix_layers 16`).
   - The router scores its own AttnRes mix (`transformer.h.{i}.router_res_query`).
   - The experts and the shared expert keep the MLP's mix (`mlp_res_query`).
   - Here: `moe_router` computes the MLP mix itself (`mlpCol`, K = 1). 1.1 takes K = 2:
     - column mlpCol: the experts' input, written to `hmlp` as today;
     - column mlpCol + 1: the router's input, scored in registers and never stored.
   - It is one dispatch, as today, with one more query column. `queryNames` puts `router_res_query` right after the
     layer's `mlp_res_query`.
2. **Layer 0 MoE.**
   - Shared expert 2048 wide: the old dense `c_fc` / `c_proj` saved as `mlp.shared_fc` / `shared_proj`, with
     `shared_expert_layer_sizes {0: 2048}`.
   - 96 x 512 routed experts, top-8, with a router mix.
   - Layer 0 leaves the dense GEMV path for `moe_router` / the fused MoE / `moe_reduce`.
   - The neuron-major buffers (`buildNeuronMajor`, `perLayer`) get a slot for layer 0, and their offsets index by
     layer, not `i - 1`.
   - The fused MoE takes the shared width as a macro (`SH = 1024 | 2048`). Only layer 0's pipelines see 2048.
3. **K/V from a mix at the prefill half's end.**
   - `kv_sources {7: 7, 16: 16, 20: 16, 24: 16, 28: 16}` and `kv_mix_layers [16]`.
   - Layer 16's own K/V project from a mix of its own (`transformer.h.16.kv_res_query`), not its query's input.
   - The `Embla.kvSources` / `bridges` constants become config: layer 7 has no bridges; layer 16 has a K/V mix row and
     bridges [20, 24, 28].
   - Layer 16's q/k/v GEMV then mixes K = 5 queries: [attn, kv16, kv20, kv24, kv28].
   - Its segments already take an input row per K/V target (`seg(..., x: m * 2048)`): the own K/V move from row 0 to
     row 1, and the bridges' rows shift by one.
   - KV jobs are unchanged.
   - Prefill: layer 16's K/V GEMM reads rows 1-4.
4. **The loop: layers [8, 16) (AttnRes blocks 2 and 3), two passes.**
   - Pass 2 re-runs the layers with their own pseudo-queries (`loop_{attn,mlp,router}_res_query`). It reads every
     source so far: emb, engram, blocks 0-3 of pass 1, then the blocks pass 2 completes.
   - Then each looped block b_j gets a second source b*_j = x1_j + loop_alpha * (x2_j - x1_j). x1 and x2 are the
     fp32 block sums of passes 1 and 2, before their bf16 store; `loop_alpha` is one [d] vector.
   - b*_j is appended after the looped blocks. Every later mix of the first pass's queries adds `loop_src_bias`
     [b_0, b*_0, b_1, b*_1] (init -log 2) to the logits of b_j and b*_j. Pass 2's own mixes take no offsets.

## The design decisions this needs

- **State slots 9 -> 12.**
  - `state` is `[tokens][9][D]` in mix.metal, prefill.metal and Prefill.swift (7 places).
  - After the loop the decode half reads emb, engram, B0-B3, B2*, B3*, B4-B6: 11 completed sources plus the partial.
  - Pass 2 itself needs emb, engram, B0-B3, B2': 7.
  - Proposal: the stride becomes a macro `NS` (9 for Embla 1, 12 for 1.1). Embla 1's bytes and dispatches are
    unchanged. Its pipelines are recompiled with NS = 9, so the same code.
- **Logit offsets.** Every mix, prep and fused-mix GEMV after the loop takes an `OFFS` macro. The offsets are read
  once per threadgroup from a [12] f32 buffer (zeros except the looped slots) and added before the softmax. Embla 1:
  OFFS = 0.
- **fp32 block sums.** Today `APPEND` writes bf16(x) into the state when a block completes. The loop's blocks (B2 at
  layer 12 in pass 1, B3 at pass 2's start, B2' at pass 2's layer 12, B3' at the end) also need x in fp32. Proposal:
  copy x (8 KB a row) to a `loopSums` buffer [4][D] f32 in the same dispatch (`KEEP` macro). b* is then one small
  kernel at the loop's end: reads 4 rows, writes 2 bf16 sources, and re-preps the columns of layer 16's block.
- **Pass 2's sliding rings.** Layers 8-15 need a second ring each: `slidingOffset` grows by 8 rings,
  `Engine.ring` x 4 kv heads x 128 x 2 x bf16. At ring = 4096 that is 4096 x 4 x 128 x 2 x 2 B = 8 MB a layer,
  64 MB in all.
- **Encoding pass 2.**
  - The step encodes layers 8-15 twice. The second time it uses the loop's query columns, its rings, and the
    pass-2 source slots.
  - The premix chain (`reduce` writing the next layer's mixes, prep at block starts) needs its own columns per pass.
    The `Layer` plans get a pass-2 twin: same weights and segments, other `attnCol` / `mlpCol` / prep, other ring
    offsets.
- **Prefill.**
  - The same twice-encoding over chunks: layers 8-15 again on the NAX GEMMs with pass-2 columns and rings, then the
    b* kernel per chunk.
  - Then layer 16's K/V mix (K = 5 rows into the K/V GEMM).
  - The prefill's mix kernels take NS / OFFS / KEEP like decode's.

## Cost

- **Decode.**
  - About +70 dispatches over ~220: pass 2 is 8 layers of mix-GEMV, attention, o-proj, router, MoE and reduce, plus
    preps, b* and the L0 router. Per dispatch the cost is what each layer's is today.
  - Weight bytes per token: +24% (the MLX port's bytes model; pass 2 re-reads layers 8-15's attention, router, shared
    expert and 8 routed experts).
  - The heimr-ios Mac proxy measured 0.80x decode for 1.1.
- **Prefill.** 8 more layers of the 16-layer prefill half, plus layer 0's MoE in place of a dense MLP: about 0.6-0.7x
  prompt tokens/s. The MLX proxy measured 0.67x.

## Check

`embla-bench check` against reference logits. heimr-edge's cached decode refuses the loop and the K/V mix (cache.py),
so 1.1's references must come from the training forward, position by position. That is a script in sparsify to add.
The function-preservation case (`--preserve`: 24 / 28 left on 7, layer 0's routed down zero, loop_alpha 0) must
reproduce Embla 1's references exactly, as it did in the MLX port.
