# escha-mlx architecture

This document describes how the runtime is put together: the layers, the data
flow from checkpoint to token, the Metal kernel surface, and the numerical
contracts that hold everything together. For *why* things are tuned the way
they are — and for every negative result — read
[BRINGUP_AND_PERF.md](BRINGUP_AND_PERF.md); for measured numbers,
[PERFORMANCE.md](PERFORMANCE.md).

## Scope

escha-mlx is a **runtime only**. It consumes finished escha checkpoints from
Hugging Face and runs them on Apple Silicon through MLX + Metal. The encoder
— calibration, trellis code search, transform construction, the recovery
fine-tune — lives outside this project and produces the checkpoints this
repository reads. The runtime's job is to reproduce the codec's numerics
**exactly** and to do it fast on memory-bandwidth-limited hardware.

Two model families are supported today, and they are different shapes of the
same codec:

| model | `model_type` | coded tensors | shape |
|---|---|---|---|
| Qwen3.6-35B-A3B MoE | `qwen3_5_moe` | expert stacks only (`gate_up_proj` K=2, `down_proj` K=3) + int8 shared expert | E-stacked streams, router-gated |
| Qwen3.8-27B dense | `qwen3_5` | all 400 projections, per-tensor rate (`mlp.{up,down}_proj` K=3, everything else K=2; 2.469 bits/weight) + int8 embed/head | single stream per matmul |

Embeddings and the LM head are never coded; they are int8 and repacked to
MLX affine-quantized Q8 at load time.

## Design principles

1. **The NumPy reference is the semantic contract.** `escha_mlx/ref.py`
   implements the codec in pure NumPy with every fp16 rounding point pinned.
   Every Metal kernel is gated bit-identical to `ref`, and `ref` is gated
   against the golden vectors committed under `tests/data/`. A kernel change
   that is not bit-identical must be measured, documented, and put behind a
   flag with the old behavior recoverable.
2. **One tree, no per-chip branches.** Every performance knob is an
   environment variable (see the [env registry](#the-env-registry-envspy)) read
   dynamically, so a strategy can be switched inside one process (the
   benchmark harnesses rely on this) and per-machine tuning is a measurement
   exercise rather than a fork.
3. **Everything is pure MLX except the kernels.** Attention, the standard KV
   caches, sampling, batching and serving all come from mlx-lm untouched.
   The custom surface is deliberately small: the coded-matmul kernels, the
   fused Hadamard transforms, the Q8 head, the GDN state kernels, and the
   MTP top-k reduction.
4. **Stream the checkpoint.** Loading is one tensor at a time from the
   safetensors shards. Holding the whole checkpoint in NumPy would peak near
   25 GB on a 24 GB machine; the streaming loader never does.
5. **Honest performance.** Every shipped default is bit-exact or has its
   numeric effect measured and documented; the two paths that reassociate the
   f32 reduction are off by default and say so in their docstrings.

## Layer map

```
┌────────────────────────────────────────────────────────────────┐
│ Interfaces                                                     │
│   generate.py (CLI) · server.py (OpenAI) · Python API          │
├────────────────────────────────────────────────────────────────┤
│ Scheduling                                                     │
│   mlx-lm BatchGenerator (AR, untouched)                        │
│   mtp_batch.py: MTPBatchGenerator + MTP{Prompt,}GenerationBatch│
├────────────────────────────────────────────────────────────────┤
│ Model — mlx-lm skeleton, escha components swapped in           │
│   EschaLinear · EschaQ8{Linear,Head,Embedding}                 │
│   EschaSparseMoeBlock · GDNStateCache · MTPHead                │
│   LastPositionHead · _FusedMLP / _FusedAttention               │
├────────────────────────────────────────────────────────────────┤
│ escha toolkits — architecture-agnostic                         │
│   quant.py (Q8 repack) · dense.py (coded linear) · moe.py      │
│   gdn_cache.py (GDN state) · mtp.py / mtp_topk.py (MTP)        │
├────────────────────────────────────────────────────────────────┤
│ Kernels — msl.py + the kernel side-modules                     │
│   mx.fast.metal_kernel → Metal; the rest is ordinary MLX ops   │
├────────────────────────────────────────────────────────────────┤
│ Contracts                                                      │
│   ref.py (bit-exact reference) · loader.py (streaming)         │
│   models/ (one plugin per model_type) · envs.py (knobs)        │
└────────────────────────────────────────────────────────────────┘
```

The dependency rule points downward: interfaces know the scheduler and the
loader; the scheduler knows the model; the model knows the toolkits; the
toolkits know the kernels and the reference. Nothing in the kernel layer
imports a model plugin, and nothing per-architecture lives below
`escha_mlx/models/`.

## Checkpoint format

An escha checkpoint is an ordinary HF safetensors export plus codec tensors.
`quantize_config.json` (falling back to `config.json → quantization_config`)
carries `quant_method`: **`escha`** (one coded stream per linear — the dense
architectures) or **`eschamoe`** (E-stacked expert streams — the MoE
architectures). `loader.is_escha_checkpoint` keys off exactly those two
strings.

Per coded projection the checkpoint stores these leaves:

| leaf | dtype / shape (dense; MoE prepends E) | meaning |
|---|---|---|
| `escha_code` | int16 `[IC/16, OC/16, 16·K]` | the packed trellis stream (K = 2 or 3) |
| `escha_rin` | fp16 `[IC]` | per-channel input scale |
| `escha_rout` | fp16 `[OC]` | per-channel output scale |
| `escha_s_in`, `escha_s_out` | f32 `[IC]`, `[OC]` | end-to-end scales (dense only; MoE exports ship all-ones and drop them) |
| `escha_config` | `[L, K, V, codebook_id, IC, OC]` | per-tensor metadata; the runtime requires `codebook_id == 1` |
| `bias` | fp16 `[OC]` | optional; applied in f32, rounded to f16 |

Dense int8 components store `weight_int8` (int8) + `weight_scale` (fp16,
one per output channel). The MTP head, where present, is a self-contained
directory: `mtp/config.json` + `mtp/model.safetensors` with exactly 15
parameter tensors, every name prefixed `mtp.`.

## The codec, precisely (`ref.py`)

The weight matrix is divided into **16×16 tiles** (256 values each). Each
tile is coded by a 2-state-per-lane trellis read off a 32-bit hash — the
production codebook (id 1). One 16×16 tile is 32 lanes × 8 decoded states,
and the lane geometry is fixed:

- **K = 2** (the gate/up legs): 16 packed u32 words per tile. Lane L reads
  words `i1 = L` and `i0 = (L+15) & 15`, merged by a per-lane bit shift.
- **K = 3** (the down leg): 24 u32 words per tile, bit offsets derived from
  `b1 = (L·8 + 257)·3`.

Each 16-bit state decodes to a half as the fp16-RNE sum of a low and a high
fp16, both taken from the hash

```
r = ((x * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60      (32-bit arithmetic)
```

`ref.decode8_k2` / `decode8_k3` do the per-lane window extraction,
`ref.lane_positions` maps lane → (row, col) inside the tile, and
`ref.reconstruct` / `reconstruct_fast` give the bare fp16 weight
(`reconstruct_fast` is vectorized and asserted bit-identical to the
reference loop). A 65 536-entry fp16 LUT of the same function
(`ref.cba_lut`) feeds the `_lut` kernel variants, which exist so the hash
path and the table path can be cross-checked and so a future codebook can be
a LUT swap.

The coded matmul is a **sandwich of unnormalized 128-point Hadamard
transforms** (Sylvester, natural order, over contiguous 128-channel blocks)
with the exact f32 constant `RS = 1/√128` pinned:

```
xh  = f16( H128( f32(x) · rin ) · RS )        # input transform
mid = f32( xh @ decode(code) )                 # the coded matmul (f32 accumulate)
y   = f16( H128( mid ) · RS · rout )           # output transform
```

For dense checkpoints `ref.fold_scales` pre-multiplies `s_in`/`s_out` into
`rin`/`rout` (one f32 product each — one rounding point *fewer* than
applying the scales separately; the deviation is documented in `ref.py`),
and the bias is added in f32 with a final f16 round. The block-level
reference `ref.moe_block` implements the whole expert block — router
softmax over top-k, the gate_up → SwiGLU → down chain, f64 accumulation of
`f32(d16) · f32(f16(score))`, and the sigmoid-gated int8 shared expert — so
the MoE path has a full reference, not just a per-matmul one.

## Load path (`loader.py`, `models/`)

`escha_mlx.load(path, load_mtp=…)` (the mlx-lm-`utils.load`-compatible
entry point; `load_model` is the model-only half) runs:

1. **Validate the environment** (`envs.validate_environment`) — fail early
   on a malformed knob, before any allocation.
2. **Apply the wired limit** if `ESCHA_MLX_WIRED_GB` is set
   (`mx.set_wired_limit`; values above the device's recommended working set
   are refused — near the cap, unwired throughput collapses ~23×).
3. **Resolve the plugin** from `config.json → model_type` via the registry
   in `models/__init__.py` (`qwen3_5`, `qwen3_5_moe`). A plugin exports
   `MODEL_TYPE` and a `CheckpointLoader` with a three-phase contract:
   - `__init__(config, group_size)` builds the mlx-lm model skeleton;
   - `consume(name, w)` is called **once per streamed tensor** with the raw
     HF name — coded leaves are grouped by base name, int8 pairs are
     repacked to Q8 on arrival, fp16 is stashed;
   - `finalize()` assembles the coded linears, runs the skeleton's
     `sanitize` over the fp16 remainder, applies post-load quirks
     (`LastPositionHead`, `gdn_cache.install`, projection fusion), and
     returns the off-tree arrays for `mx.eval`.
4. **Stream the shards** — `glob` of `*.safetensors` in order,
   `safe_open(framework="numpy")`, one `get_tensor` at a time.
5. **MTP** — only `qwen3_5` may carry an MTP head; `mtp.load_mtp` reads the
   `mtp/` directory, shape-checks every tensor against the head, and raises
   on anything unexpected or missing.

The plugins are where architecture-specific knowledge lives, and the split
is strict: the codec engine (`ref`/`msl`/`quant`/`dense`/`moe`) never forks
per architecture; each `model_type` contributes one plugin.

- **`models/qwen3_5.py` (dense)** installs `EschaLinear` on every projection
  (`mlp.{gate,up,down}_proj`, `self_attn.{q,k,v,o}_proj`,
  `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`), wraps `lm_head`/
  `embed_tokens` in the Q8 classes, and — via `install_projection_fusion` —
  installs `_FusedMLP` (gate+up through `dense.project_pair` in one sharded
  dispatch, SwiGLU, down) and `_FusedAttention` (K/V paired; the Qwen3.5
  gated attention splits q into `queries, gate` and applies
  `o_proj(out · sigmoid(gate))`).
- **`models/qwen3_5_moe.py`** replaces each layer's MLP with
  `EschaSparseMoeBlock`: an f16 `nn.Linear` router (f32 logits, top-k via
  `argpartition`, `softmax(…, precise=True)`), the `EschaExperts` E-stacked
  code arrays, and a fixed-order segmented sum for the score-weighted
  combine (replacing an atomic scatter-add, for determinism). The shared
  expert is int8-Q8 with the sigmoid gate.

## Forward pass

### A coded dense linear (`dense.py`)

`EschaWeight` holds the packed u32 code (`[TK, TN, 8K]` — raw int16
`[TK, TN, 16K]` is converted on arrival), the folded `rin`/`rout` (f32),
and validates the metadata (K ∈ {2,3}; IC/OC multiples of 128;
`codebook_id == 1`). `EschaLinear.__call__` runs the sandwich above, with
the middle leg dispatched by row count `r`:

| `r` | kernel | notes |
|---|---|---|
| 1 | `escha_gemv_dense_k{K}` (direct) | the default per-row path |
| 2–8 | `escha_gemm_dense_direct_k{K}_r{R}` | barrier-free small-R GEMM; with the output transform fused in the epilogue (`_direct_out`) when possible |
| > 8 | `escha_gemm_dense_k{K}_r{R}_kb{KB}` | row-blocked, staged; R is the power-of-two block snapped from `min(m, 16)` |
| 16 (opt-in) | `escha_gemm_dense_sgmat_k{K}` | `simdgroup` matrix multiplies; **deterministic, not bit-identical** — off by default |

`dense.project_pair` evaluates two coded linears (gate+up, K/V, qkv+z) in
one sharded dispatch and falls back to two separate calls when the shapes
don't line up. The input/output Hadamard legs are either the fused Metal
kernels (default) or the ordinary MLX `mx.hadamard_transform` ops path
(`ESCHA_MLX_FUSED_HAD=0`).

### An MoE expert block (`moe.py`)

Rows are (token, slot) pairs in token-major order, `row_expert` i32. The
block: input transform → per-row GEMV (decode) or **row-blocked GEMM**,
where `moe.build_groups` stable-argsorts the rows and scatters them into
fixed-R groups sharing one expert — R rows against the same decoded weight
stream, one code fetch per group (bit-identical to the R=1 kernel) →
output transform → SwiGLU in f16 → the K=3 down leg → score weighting →
fixed-order segmented sum in f64 → f16 → shared-expert add. The row-block
size comes from a size policy (`_blocked_R`: 12 / 4 / 3 / 2 / 1 at
m ≥ 2048 / 1024 / 320 / 128 / else) or a pinned `ESCHA_MLX_BLOCK_R`.

### The int8 Q8 repack (`quant.py`)

`pack_q8` turns `weight_int8 [N, K]` + per-channel f16 `weight_scale` into
MLX affine-quantized inputs: the codes are `w8 XOR 0x80` packed little-
endian into u32 `[N, K/4]`, with f32 scales and biases (`bias = −128·scale`)
repeated per group (default group 128; 64 and 32 supported, auto-degraded
when K doesn't divide). Scales/biases are f32 on purpose — f16 would give
half-precision dequant on the CPU backend. `EschaQ8Linear` then runs
`mx.quantized_matmul(…, group_size, bits=8)`. A one-time
`validate_pack` round-trips the repack on-device and **refuses to serve**
on mismatch. `EschaQ8Head` uses a dedicated Metal kernel
(`escha_q8_head_r{rows}`) for the 1–8-row decode case and falls back to the
MLX op otherwise.

## The Metal kernel layer (`msl.py`)

All kernels are `mx.fast.metal_kernel` with MSL generated by string
templating; MoE and dense are a **compile-time variant** — the only
structural difference is how the code stream and scale offsets are indexed
(expert indirection present or absent). Kernel names encode every source
variant, since MLX's compiled-kernel cache identity is the name. The
families:

| family | kernels (name prefixes) | computes |
|---|---|---|
| Tile decode | `escha_decode_tiles_k{2,3}` (+ `_lut`) | one 16×16 tile → bare fp16 weights; one simdgroup per tile, 32 lanes × 8 states |
| Fused Hadamard | `escha_scaled_had`, `escha_scaled_had_out` (+ `_dense`, `_pair`) | the sandwich legs: `f16(H128(·)·RS·scale)` in one dispatch, 128-thread threadgroup per 128-channel block; the two post-transform f32 multiplies stay separate statements to preserve both rounding points |
| GEMV (per row) | `escha_moe_gemv_k{K}` / `escha_gemv_dense_k{K}` (staged); `…_direct` (+ `_shf`, `_pf{n}`, `_sk{S}` variants) | `xh[r] @ decode(code[row_expert[r]])`; staged version keeps 8 code words + 16 x values in threadgroup memory (two barriers per k-tile); **direct** (the default) is barrier-free, each simdgroup reading its own code tile; `_pf` hoists n code fetches into registers; `_sk` splits the k-chain (deterministic, not bit-identical — `mx.sum` over partials in fixed order) |
| Row-blocked GEMM | `escha_moe_gemm_k{K}_r{R}_kb{KB}` (+ `_sx`, `_pf`, `_lut`); dense `escha_gemm_dense_direct_k{K}_r{R}` (+ `_out`), `escha_gemm_dense_pair[_out]_k{K0}{K1}_r{R}` | R rows sharing one expert decode share one code stream; `kb` k-tiles of x per barrier pair; the dense `_direct` variant is barrier-free for R ≤ 8 (the MTP M=4 verification shapes); `_out` fuses the output Hadamard into the epilogue; `_pair` runs two coded projections (possibly different K) in one dispatch |
| simdgroup-matrix GEMM | `escha_gemm_dense_sgmat_k{K}` (+ `_lut`) | opt-in R=16 dense GEMM on `simdgroup_multiply_accumulate`; reassociates the f32 reduction — off by default |

Dispatch policy — `use_direct`, `use_shuffle`, `split_k_for`, `kt_block`,
`use_prefetch`, `use_sortx`, `dense_block_r`, `use_dense_mat` — reads the
env registry and decides which compiled variant runs. Split-K is the only
policy that can change numerics (it changes the summation order of the f32
accumulate); it is off by default, its `auto` mode only engages below 512
threadgroups, and the change is documented as deterministic-but-not-
bit-identical with a reproducibility gate.

The three kernel side-modules keep `msl.py` about the codec:

- **`quant.py`** — the `escha_q8_head_r{rows}` decode kernel (one simdgroup
  per output channel, lane-strided over the packed u32s).
- **`mtp_topk.py`** — a two-stage top-3 + logsumexp reduction over the
  vocabulary: `escha_mtp_topk_partial_k3` reduces 2048-logit blocks
  (per-lane top-3 scan → threadgroup candidate merge → block summary), and
  `escha_mtp_topk_final_k3` merges the block summaries with
  `log_norm = global_max + log(total_sum)`. Falls back to MLX
  `argpartition` for small vocabularies, non-f16 inputs, or unsupported
  shapes; the logsumexp association may differ by one fp16 ulp, which is
  safe because every proposed token is target-verified.
- **`gdn_cache.py`** — the GDN kernels, below.

## Gated-DeltaNet state (`gdn_cache.py`)

Both Qwen3.5/3.8 checkpoints interleave GDN (linear-attention) layers with
full-attention layers. mlx-lm keeps the GDN recurrent state — slot 1 of the
`ArraysCache`, `[B, heads, Dv, Dk]` (multi-MB per layer) — in f32.
escha-mlx:

- **`GDNStateCache`** casts only slot 1 to the dtype chosen by
  `ESCHA_MLX_GDN_STATE` (default `fp16`; `bf16` available; `fp32` is a
  genuine no-op that restores the stock mlx-lm cache exactly). The cache
  stores the dtype *name* string, because `mx.Dtype` is not picklable and
  the batch generator deep-copies prompt caches.
- **Allocation-free first state.** The first forward of a fresh cache would
  otherwise allocate the full f32 zero state transiently.
  `escha_gated_delta_zero{,_mask}` runs the T-token recurrence in f32
  registers initialized to zero and writes the state directly in the storage
  dtype — bit-identical, ~2 GB of peak memory saved at batch 64 (measured in
  [PERFORMANCE.md](PERFORMANCE.md)).
- **MTP state traces.** Tree verification needs each node's state, and each
  node branches from its *parent's* state:
  - `escha_gated_delta_trace{,_mask}` — recurrence from a given state,
    emitting `state_trace [B, T, Hv, Dv, Dk]` after every token;
  - `escha_gated_delta_tree_trace` — each tree node starts from its parent
    node's state (`parents` i32; −1 = committed base state);
  - `escha_gdn_tree_conv_windows` — builds all parent-relative
    depthwise-convolution windows in one pass.
  `trace_gdn_forward` runs one GDN layer explicitly (sharing one
  `project_pair` dispatch for the qkv/z projections), and
  `commit_tree_states` commits each request's accepted node's conv +
  recurrent state via `take_along_axis`. All have portable NumPy fallbacks
  for the CPU test path.

## Multi-token prediction (`mtp.py`, `mtp_batch.py`)

The Qwen3.8 checkpoint ships a **native MTP head**: one full-attention
decoder layer (`mtp_num_hidden_layers == 1`, shared embeddings required)
with the alignment `(hidden_t, embedding_{t+1}) → token_{t+2}`.
`load_mtp` reads its 15 tensors and shares the target's embedding and LM
head by reference, so they stay out of the MTP parameter tree.

**Proposal.** `_propose_tree` builds a fixed lattice from the seed logits:
level 0 = top-3 (via the `mtp_topk` kernel or the `argpartition`
fallback); each of the next `depth − 1` levels expands the frontier k²
ways. That is `3 + 9 + 9 = 21` candidate paths, ranked by cumulative log
probability; the best ancestor-closed `draft_nodes = 3` are kept in
depth-major topological order. The proposal levels are appended
**temporarily** to the committed MTP KV cache (branches share one physical
prefix) and trimmed in a `finally` — never `batch × topk` copies.

**Verification.** `_target_tree_forward` runs the target model over the
M = 4 rows (root + 3 drafts): full-attention layers get an ancestor mask and
depth-aware RoPE against the shared prefix; GDN layers run
`trace_gdn_forward` with `tree_parents`. Target tokens are sampled per row
(with each request's own sampler in the batched path), the accepted chain
is materialized, and then:

- attention histories are **right-aligned** into the rectangular
  `BatchKVCache` (per-request padding, offsets updated),
- GDN/conv state is committed from that request's final accepted node,
- the MTP head replays the accepted suffix with **replayed hiddens**
  (`mx.take` of the verification hiddens) — no target re-forward.

So every request in a continuous batch commits its *own* path; a
low-acceptance request no longer shortens the batch. When only one token
remains, the loop drops to one plain target forward (no tree).

**Continuous batching** (`mtp_batch.py`) subclasses the mlx-lm 0.31 batch
scheduler:

- `MTPBatchGenerator` — inherits the stock `BatchGenerator`; **rejects
  `max_kv_size`/rotating caches at setup**; its `prompt_cache_nbytes`
  accounts for both MTP caches so the prompt-cache budget sees them.
- `MTPPromptProcessingBatch` — chunked target prefill with **online MTP
  prefill**: the MTP head consumes the target's hidden states as they are
  produced (pairing `H(x_t)` with `x_{t+1}`), so a fresh request needs no
  separate MTP pass; a target-only prompt-cache hit falls back to one
  replay, because that public cache format has no historical hiddens.
- `MTPGenerationBatch` — the per-round machinery above, plus per-request
  sampling, early-stop extraction (`_finished_target_cache` trims the
  attention nodes and rewinds GDN state to the exact stop position inside
  an accepted burst), and the memory discipline that releases the previous
  GDN trace before building the next verification graph (retaining two
  traces pushes an MTP B=8 server over the Metal working-set limit).

The server applies the conservative defaults that follow from the measured
crossover (tree verification wins at B = 1/2, loses to batched AR from
B = 4): `--decode-concurrency 2`, `--prompt-concurrency 2`, and a 512 MB
prompt-cache budget (`_MTP_DEFAULT_PROMPT_CACHE_BYTES`), all overridable.

## Serving and CLI

- **`generate.py`** — one-shot CLI. Loads through `escha_mlx.load`, and
  generates through `mtp.stream_generate` (which tags each emitted token
  `from_draft`) when `--mtp` is set, otherwise mlx-lm's own
  `stream_generate`. Chat-template handling is tri-state: `enable_thinking`
  / `reasoning_effort` are only passed when explicitly requested, because
  the Qwen3.8 template renders an omitted `enable_thinking` as *true*.
- **`server.py`** — a thin wrapper over `mlx_lm.server` (continuous
  batching, prefix caching, reasoning parsing, tool calls all come from
  mlx-lm). It monkeypatches `server.load` to route escha checkpoints through
  `escha_mlx.load`, and, when `--mtp` is set, swaps exactly one scheduler
  class: `server.BatchGenerator = MTPBatchGenerator`. Two local extensions:
  an `"ignore_eos": true` request flag (a sentinel stop word injected into
  the response state machine) and `GET/POST /v1/cache/clear` (releases the
  Metal allocator and reports bytes freed — the escape hatch for the
  `iogpu.rsrc_limit` buffer-count failure mode). Structured-output /
  `json_schema` constrained decoding is a known gap.

## The env registry (`envs.py`)

Every knob in the package is declared once in `envs.py` — the *only* module
that reads the environment — as a typed `EnvVar` with a layer
(`BUILD`, `DEPLOYMENT`, `RUNTIME`, `KERNEL`, `DEVELOPMENT`) and a default.
Values are **re-read on every access**, which is what lets the benchmark
harnesses flip strategies inside one process. Highlights:

| variable | default | effect |
|---|---|---|
| `ESCHA_MLX_WIRED_GB` | off | wire the working set before allocation; refuse values above the device's recommended cap |
| `ESCHA_MLX_GDN_STATE` | `fp16` | GDN recurrent-state dtype; `fp32` restores the stock cache exactly |
| `ESCHA_MLX_Q8_GROUP` | `128` | Q8 repack group size (128/64/32) |
| `ESCHA_MLX_LAST_LOGIT` | `true` | slice the LM head to the last position in prefill |
| `ESCHA_MLX_BIAS` | off | apply the per-linear correction biases a dense export ships beside its coded linears. Off by default: every published number for the shipped checkpoints was produced without them, so the runtime reproduces the published model; the export's written contract includes them — see the `dense.py` docstring before flipping either way |
| `ESCHA_MLX_DENSE` | `q8` | `fp16` falls back to ordinary `nn.Linear`/`nn.Embedding` |
| `ESCHA_MLX_MOE`, `ESCHA_MLX_LINEAR` | `fused` | `ops` selects the ordinary-MLX path (the reference implementation of the same math) |
| `ESCHA_MLX_FUSED_HAD` | `true` | fused Hadamard kernels vs `mx.hadamard_transform` |
| `ESCHA_MLX_GEMV`, `ESCHA_MLX_FETCH`, `ESCHA_MLX_PREFETCH`, `ESCHA_MLX_SORTX`, `ESCHA_MLX_SPLITK`, `ESCHA_MLX_KT_BLOCK`, `ESCHA_MLX_BLOCK_R` | `direct` / `load` / off / off / `1` / `4` / policy | GEMV/GEMM micro-strategies; `SPLITK` changes numerics (documented), the rest are bit-exact |
| `ESCHA_MLX_LUT` | off | LUT decode kernels instead of the inline hash |
| `ESCHA_MLX_DENSE_MAT`, `ESCHA_MLX_DENSE_BLOCK_R`, `ESCHA_MLX_GEMV_PF`, `ESCHA_MLX_Q8_HEAD` | off / policy / `1` / on | dense GEMM strategy, block pinning, fetch prefetch, Q8 head kernel |
| `ESCHA_MODEL`, `ESCHA_DENSE_MODEL`, `ESCHA_MLX_SLOW_TESTS` | unset | real-checkpoint test gates (see below) |

## Numerical contracts and determinism

The test pyramid, in order of trust:

1. **Goldens** — committed reference vectors under `tests/data/`: the
   codec (k2 and k3 packed tiles → expected fp16 weights), the w8a16
   repack, a 128×128 corner of a shipped dense linear (all leaves, including
   `s_in`/`s_out`/`bias` → expected output), and a full MoE block
   (input + router ids/scores → expected output). `tests/conftest.py` is the
   only consumer.
2. **`ref.py`** — the NumPy reference, gated against the goldens on any
   platform (this is the part that runs on Linux CI with `mlx[cpu]`).
3. **Metal kernels** — gated **bit-identical** to `ref` and to each other
   (staged vs direct GEMV, hash vs LUT decode, row-blocked vs per-row GEMM,
   fused vs ops Hadamard, native `mx.hadamard_transform` order). The Metal
   gates self-skip when the runner has no Metal, so the same suite is
   runnable on Linux for the reference/repack half.
4. **End-to-end** — real-checkpoint tests gated by `ESCHA_MODEL` /
   `ESCHA_DENSE_MODEL`, heavy ones behind `ESCHA_MLX_SLOW_TESTS`.

The shipped-default paths are all bit-exact. The deliberately non-bit-
identical paths, each measured and each with a documented fallback or
opt-out:

| path | status | why it's safe |
|---|---|---|
| split-K GEMV (`_sk`) | off by default; deterministic | changes f32 accumulate order; reproducibility + rounding-agreement gates |
| `simdgroup`-matrix dense GEMM (`ESCHA_MLX_DENSE_MAT`) | off by default; deterministic | reassociates the f32 reduction; measured +28% prefill on the M5 Pro, documented in [PERFORMANCE.md](PERFORMANCE.md) |
| MoE segmented sum (vs atomic scatter-add) | shipped default | f64 fixed-order sum; the ops path (`ESCHA_MLX_MOE=ops`) is the reference it's gated against |
| Q8 repack | shipped default | ≤ 0.5-ulp f16 deviation vs the int8 reference; `validate_pack` round-trips it on-device at load and refuses to serve on mismatch |
| `fold_scales` | shipped default (dense) | one f32 rounding point fewer than applying `s_in`/`s_out` separately; documented in `ref.py` |
| MTP top-k logsumexp | shipped default | may differ by one fp16 ulp from MLX; proposal scoring only — every token is target-verified |

The skip/gate mechanism is capability-based and import-time: `needs_mlx`,
`needs_metal`, `needs_ckpt`, `needs_dense_ckpt`, `needs_slow` in
`tests/conftest.py`. `pytest tests/` is safe anywhere; on a Mac with the
checkpoints present, everything runs.

## Testing and benchmarking layout

- **`tests/`** — `test_ref_decode.py` (codec, pure NumPy), `test_metal.py`
  + `test_gemv_direct.py` + `test_splitk.py` + `test_blocked.py` +
  `test_fused_had.py` + `test_quant_metal.py` (kernel bit-exactness),
  `test_dense_linear.py` (the dense path, 43 test functions, including the
  cross-runtime gate on the shipped-linear golden), `test_mlx_cpu.py` (Q8
  repack + MoE ops path vs the MoE-block golden), `test_moe_determinism.py`,
  `test_gdn_state_dtype.py`, `test_models.py` (registry + synthetic
  checkpoints, MoE and dense), `test_last_logit.py`, `test_envs.py`,
  `test_benchmark_metadata.py`, `test_dense_checkpoint.py` (real-export
  structural contract), and `test_mtp.py` (the MTP head, tree masks,
  greedy AR-vs-MTP equivalence single-stream and batched, per-request path
  commit, early-stop extraction).
- **`bench/`** — `p0_gates.py` is the first thing to run on a new machine
  (bit-exact decode, measured DRAM GB/s, Q8 round-trip, memory envelope);
  `roofline.py` bounds tok/s from measured bandwidth; `isl_osl_grid.py`
  drives the OpenAI endpoint; `head_to_head.py`, `sweep_*.py` do paired
  in-model A/B with drift controls; `mtp.py` reproduces the MTP numbers.
  Results live per machine under `bench/results/<machine>/` — the format and
  the method are specified in [CONTRIBUTING.md](../CONTRIBUTING.md) and
  [BRINGUP_AND_PERF.md](BRINGUP_AND_PERF.md).

## Extending

- **New architecture.** Add a plugin under `escha_mlx/models/` (skeleton +
  tensor map + quirks), register its `model_type` in the `REGISTRY`, supply
  goldens, then wire the kernels — the codec engine is shared as-is. The
  plugin contract is documented in `models/__init__.py`.
- **New kernel strategy.** Add the compiled variant (name it after its
  source), a dispatch helper reading the env registry, a bit-exactness
  test against the existing paths, and a bench A/B. If it changes numerics,
  it stays off by default until measured and documented.
- **New chip.** No branches: run `bench/p0_gates.py`, set the knobs that
  win, and land the numbers under `bench/results/<machine>/`.

## Repository layout

| path | what |
|---|---|
| `escha_mlx/ref.py` | NumPy bit-exact codec reference — the semantic contract |
| `escha_mlx/msl.py` | the Metal kernels: decode, Hadamard transforms, GEMV ×2, row-blocked GEMM, fused variants, sgmat GEMM |
| `escha_mlx/quant.py` | int8→Q8 repack + Q8 linear/head/embedding (one Metal kernel) |
| `escha_mlx/dense.py` | dense coded-linear toolkit (`EschaWeight`, `EschaLinear`, `project_pair`) |
| `escha_mlx/moe.py` | expert toolkit (`EschaExperts`, router group building, Hadamard blocks) |
| `escha_mlx/loader.py` | streaming loader, checkpoint detection, wired limit, plugin dispatch |
| `escha_mlx/models/` | one plugin per `model_type` (`qwen3_5`, `qwen3_5_moe`) |
| `escha_mlx/gdn_cache.py` | GDN state dtype, allocation-free first state, MTP state-trace kernels |
| `escha_mlx/mtp.py` | native MTP head, 15-tensor loader, proposal tree, speculative verify loop |
| `escha_mlx/mtp_topk.py` | fused Metal top-3 / logsumexp reduction |
| `escha_mlx/mtp_batch.py` | native-MTP continuous batching with per-request tree-path commit |
| `escha_mlx/envs.py` | the single env-var registry (typed, layered, dynamically re-read) |
| `escha_mlx/benchmark_metadata.py` | runtime/model revision metadata for benchmark reports |
| `escha_mlx/{generate,server}.py` | CLI / OpenAI-compatible server |
| `tests/` + `tests/data/` | golden-gated suite + the committed reference vectors |
| `bench/` | gates, roofline, serving grid, sweeps; results per machine under `bench/results/` |
| `docs/` | install · tuning · performance · this document |
