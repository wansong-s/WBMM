<h1 align="center">WBMM</h1>

<p align="center">
  <b>Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.02097">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  &nbsp;
  <a href="https://arxiv.org/pdf/2607.02097">
    <img src="https://img.shields.io/badge/PDF-Download-b31b1b?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="PDF">
  </a>
  &nbsp;
  <a href="https://github.com/wansong-s/WBMM">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="Code">
  </a>
  &nbsp;
  <a href="https://huggingface.co/wansong-s/WBMM/tree/main">
    <img src="https://img.shields.io/badge/Weights-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=white" alt="Weights">
  </a>
  &nbsp;
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-4c8eda?style=flat-square" alt="License">
  </a>
</p>

> 🎉 **This work has been accepted to ICML 2026 as a Spotlight paper.**

WBMM replaces the expensive large-kernel depth-wise convolution with a
**windowed batched matrix multiplication**: inside each local window the feature
map is multiplied by a position-dependent matrix built from a small
relative-position table. This gives a large effective receptive field at a
fraction of the cost — and, unlike most large-kernel methods, it needs **no
custom CUDA / cutlass / iGEMM extension**; it runs anywhere ordinary
`torch.matmul` runs.

> **Fairness.** The classification, detection and segmentation training recipes
> are kept **identical to [UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet)**.
> Every other factor (macro-design, augmentation, optimizer, schedule) is held
> constant — only the operator changes — so any difference in accuracy or speed
> is attributable to the **WBMM operator alone**. This repo is built on the
> UniRepLKNet codebase (Apache-2.0).

## Notation: `W` and `D`

A per-block operator is written exactly as in the paper:

| token | meaning |
|:-----:|:--------|
| `W`   | a **WBMM** block (the windowed batch-matmul operator) |
| `D`   | a **3×3 depth-wise** convolution |

So `('W', 'D', 'W')` means *WBMM → 3×3 DW → WBMM*.

**The window size is a parameter** — `window_size=(7, 7)` — and **defaults to
7×7**, the setting used for every released model. The operator is fully
window-agnostic, e.g. `wbmm_t(window_size=(w, w))`; the dense backbones zero-pad
internally so they accept any input resolution.

## Two model families

The classification models and the dense-prediction backbones differ in a
**single place — the first stage `S1`**:

| family | `S1` pattern | used for |
|:-------|:-------------|:---------|
| classification | `[D, D, D]` (all 3×3) | reporting ImageNet Top-1 |
| dense backbone | `[W, D, W]` (mixes WBMM) | detection / segmentation |

Everything after `S1` is identical. The dense `S1=[W,D,W]` trades a little
ImageNet Top-1 for **stronger downstream features**, so we release those
ImageNet checkpoints (they are what the det/seg configs initialise from) but do
not report their Top-1.

## Model Zoo

All checkpoints live in [`wansong-s/WBMM`](https://huggingface.co/wansong-s/WBMM/tree/main).
With `huggingface_hub` installed the code can download them automatically
(`in_1k_pretrained=True`).

> **Naming convention.** Every released checkpoint — classification, dense
> backbone, detection and segmentation alike — carries an explicit window tag
> in its file name. All current releases use the **7×7 window** (the standard
> setting for this repo), hence the `_w7` suffix; a future release with a
> different window would be tagged accordingly, e.g. `..._w9.pth`. The `window`
> column below restates this for each model.

### ImageNet-1K classification

| model | window | #params | FLOPs | Top-1 | checkpoint |
|:------|:------:|:-------:|:-----:|:-----:|:-----------|
| WBMM-P | 7×7 | 10.6M | 1.6G | **80.3** | [wbmm_p_in1k_224_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_p_in1k_224_w7.pth) |
| WBMM-N | 7×7 | 18.1M | 2.7G | **81.7** | [wbmm_n_in1k_224_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_n_in1k_224_w7.pth) |
| WBMM-T | 7×7 | 31.0M | 4.8G | **83.2** | [wbmm_t_in1k_224_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_t_in1k_224_w7.pth) |
| WBMM-S | 7×7 | 55.6M | 9.0G | **83.9** | [wbmm_s_in1k_224_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_s_in1k_224_w7.pth) |

WBMM is **1.31×–1.88× faster** than the matched UniRepLKNet models.

### Dense-task ImageNet backbones (`S1=[W,D,W]`)

Used to initialise the detection / segmentation models below.

| model | window | checkpoint |
|:------|:------:|:-----------|
| WBMM-T (dense) | 7×7 | [wbmm_t_in1k_224_dense_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_t_in1k_224_dense_w7.pth) |
| WBMM-S (dense) | 7×7 | [wbmm_s_in1k_224_dense_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_s_in1k_224_dense_w7.pth) |

### ADE20K segmentation — UPerNet, 160k

| backbone | window | crop | #params | FLOPs | mIoU (SS) | mIoU (MS) | config | checkpoint |
|:---------|:------:|:----:|:-------:|:-----:|:---------:|:---------:|:-------|:-----------|
| WBMM-T | 7×7 | 512 | 62M | 944G | 48.3 | 48.8 | [config](segmentation/configs/ade20k/upernet_wbmm_t_512_160k_ade20k.py) | [upernet_wbmm_t_ade20k_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/upernet_wbmm_t_ade20k_w7.pth) |
| WBMM-S | 7×7 | 512 | 87M | 1033G | 50.2 | 50.5 | [config](segmentation/configs/ade20k/upernet_wbmm_s_512_160k_ade20k.py) | [upernet_wbmm_s_ade20k_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/upernet_wbmm_s_ade20k_w7.pth) |

### COCO detection — Cascade Mask R-CNN, 3×

| backbone | window | #params | FLOPs | AP<sup>box</sup> | AP<sup>mask</sup> | config | checkpoint |
|:---------|:------:|:-------:|:-----:|:----:|:----:|:-------|:-----------|
| WBMM-T | 7×7 | 89M | 747G | 51.6 | 44.8 | [config](detection/configs/coco/casc_mask_rcnn_wbmm_t_in1k_fpn_3x_coco.py) | [casc_mask_rcnn_wbmm_t_coco_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/casc_mask_rcnn_wbmm_t_coco_w7.pth) |
| WBMM-S | 7×7 | 113M | 833G | 52.8 | 45.6 | [config](detection/configs/coco/casc_mask_rcnn_wbmm_s_in1k_fpn_3x_coco.py) | [casc_mask_rcnn_wbmm_s_coco_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/casc_mask_rcnn_wbmm_s_coco_w7.pth) |

## Installation (image classification)

```bash
git clone https://github.com/wansong-s/WBMM.git
cd WBMM

conda create -n wbmm python=3.8 -y
conda activate wbmm

# install a torch build that matches your CUDA, e.g. CUDA 11.3:
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

Detection / segmentation have extra MM-series dependencies — see
[detection/README.md](detection/README.md) and
[segmentation/README.md](segmentation/README.md).

## ImageNet evaluation

```bash
# single GPU
python main.py --model wbmm_t --eval true \
  --resume wbmm_t_in1k_224_w7.pth --input_size 224 \
  --data_path /path/to/imagenet-1k

# 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wbmm_t --eval true \
  --resume wbmm_t_in1k_224_w7.pth --input_size 224 \
  --data_path /path/to/imagenet-1k
```

> The commands above evaluate the **original, un-fused** checkpoint only —
> `main.py` builds its model from [`wbmm.py`](wbmm.py) and calls a strict
> `model.load_state_dict()`, so it cannot load a *fused* checkpoint (its keys
> differ: `R_fused`/`bias_fused` or `M_fused`/`bias_fused` instead of
> `relative_position_bias_table` + BatchNorm stats). To get real Top-1/Top-5
> for the two fused checkpoints `reparameterize.py` produces
> (`deploy_nocache` and `deploy_cache`), use [`eval_deploy.py`](eval_deploy.py)
> instead — see **"Evaluate the fused checkpoints"** under
> "Reparameterization for inference" below.

## ImageNet-1K training

The recipe is **identical to UniRepLKNet** (that is the point — see the fairness
note). We use an initial learning rate of `4e-3` and a **total batch size of
4096**, i.e. `num_gpus × batch_size × update_freq = 4096`. Per-model
`drop_path` / `mixup` / `cutmix` match UniRepLKNet exactly.

> If you hit OOM, reduce `--batch_size` and increase `--update_freq`
> proportionally to keep the total batch at 4096 (e.g. on 4 GPUs:
> `--batch_size 64 --update_freq 16`).

### 4-GPU (up to Tiny)

The smaller models (P / N / T) were trained on **4 GPUs** in UniRepLKNet; the
same commands apply here. (Small requires 8 GPUs — see below.)

**WBMM-P**
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --model wbmm_p --drop_path 0.1 \
  --batch_size 128 --lr 4e-3 --update_freq 8 \
  --mixup 0.3 --cutmix 0.3 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

**WBMM-N**
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --model wbmm_n --drop_path 0.1 \
  --batch_size 128 --lr 4e-3 --update_freq 8 \
  --mixup 0.5 --cutmix 0.5 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

**WBMM-T**
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --model wbmm_t --drop_path 0.2 \
  --batch_size 128 --lr 4e-3 --update_freq 8 \
  --mixup 0.8 --cutmix 1.0 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

### 8-GPU (all sizes)

**WBMM-P**
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wbmm_p --drop_path 0.1 \
  --batch_size 64 --lr 4e-3 --update_freq 8 \
  --mixup 0.3 --cutmix 0.3 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

**WBMM-N**
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wbmm_n --drop_path 0.1 \
  --batch_size 64 --lr 4e-3 --update_freq 8 \
  --mixup 0.5 --cutmix 0.5 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

**WBMM-T**
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wbmm_t --drop_path 0.2 \
  --batch_size 64 --lr 4e-3 --update_freq 8 \
  --mixup 0.8 --cutmix 1.0 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

**WBMM-S**
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wbmm_s --drop_path 0.4 \
  --batch_size 64 --lr 4e-3 --update_freq 8 \
  --mixup 0.8 --cutmix 1.0 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

> The dense-task backbones use the **same recipe** as their counterpart — train
> `wbmm_t_dense` exactly like `wbmm_t`, and `wbmm_s_dense` like `wbmm_s` (only
> the model name changes).

## Using the WBMM operator on its own

```python
import torch
from wbmm import wbmm           # the operator (7x7 by default)

op = wbmm(dim=320, window_size=(7, 7))
x  = torch.randn(2, 320, 28, 28)
y  = op(x)                      # same shape

from wbmm import wbmm_t
model = wbmm_t(in_1k_pretrained=True)   # auto-downloads wbmm_t_in1k_224_w7.pth from HF
```

## Reparameterization for inference (BatchNorm / multi-kernel fusion)

> **New files, unchanged training.** The fusion machinery below lives in
> three new files: [`wbmm_reparam.py`](wbmm_reparam.py),
> [`reparameterize.py`](reparameterize.py), and
> [`eval_deploy.py`](eval_deploy.py) (real-accuracy evaluation for the fused
> checkpoints — see "Evaluate the fused checkpoints" below).
> [`wbmm.py`](wbmm.py) and `main.py` are untouched — training, and the
> "ImageNet evaluation" commands above, still work exactly as before.
> `wbmm_reparam.py` is `wbmm.py` plus this inference-time machinery layered
> on top (identical trainable parameters, same names and shapes), so it
> loads any checkpoint `main.py` produces without modification — see
> [`tests/test_equivalence.py`](tests/test_equivalence.py) for the numerical
> proof. The two model files register the same timm model names, so import
> only one of them per process (`wbmm_reparam.py`'s own module docstring
> explains why; `eval_deploy.py` and `reparameterize.py` both already follow
> this rule by only ever importing `wbmm_reparam`).

Every WBMM ('W') block is wrapped in a BatchNorm (`WBMMBlock.norm`); the S4
stage of WBMM-P / WBMM-N additionally fuses two parallel depthwise paths
(`Sec. 3.7`: `BN1(WBMM(x)+x) + BN2(DW5(x)) + BN3(DW3(x))`); and every block's
FFN projection (`pwconv2`) ends in a BatchNorm too. At inference, all of this
is *pure linear algebra applied to a fixed input* — no batch statistics are
computed — so it can be folded, once and for all, directly into the WBMM
table/matrix / conv kernel / linear weights it feeds. `reparameterize_wbmm()`
methods (added to `wbmm`, `WBMMBlock` in `wbmm_reparam.py`) do exactly this,
into **one of two interchangeable deploy targets** you choose with a `cache`
flag.

Shapes used below (`C` = channels, `d = wh*ww` = elements per window): `I` is
the `(d, d)` `relative_position_index` buffer (see
`wbmm._get_relative_position_index`) — **not** flattened — so `R_fused[:, I]`
is ordinary PyTorch/NumPy advanced indexing that already returns the dense
`(C, d, d)` matrix, identical to `torch.index_select(R_fused, 1,
I.flatten()).view(C, d, d)` (exactly what `wbmm._build_matrix` computes; both
forms are equal element-for-element). `bias_fused` is `(C,)` and must
broadcast as a per-channel scalar over the window-batched `(C, N, d)` input
(`N` = batch × number of windows) — i.e. `bias_fused[:, None, None]`
(`.view(C, 1, 1)` in the code) — a bare `+ bias_fused` does **not** broadcast
against a `(C, N, d)` tensor and raises a shape-mismatch error, so the tables
below always show it reshaped:

| before fusion | after fusion, default (`cache=False`) | after fusion, `--cache` (`cache=True`) |
|:--------------|:---------------------------------------|:-----------------------------------------|
| `BN1(WBMM(x)+x) + BN2(DW5(x)) + BN3(DW3(x))` (S4 of P/N) | `x @ R_fused[:, I] + bias_fused[:, None, None]` | `x @ M_fused + bias_fused[:, None, None]` |
| `BN( WBMM(x) + x )` (any other 'W' block)                | `x @ R_fused[:, I] + bias_fused[:, None, None]` | `x @ M_fused + bias_fused[:, None, None]` |
| `BN( DWConv(x) )` ('D' blocks)                            | `Conv2d(x)` (bias folded in)     | *(same — no WBMM operator here)* |
| `BN( Linear(x) )` (`pwconv2`, every block)                | `Linear(x)` (bias folded in)     | *(same)* |

(`x` itself is reshaped/permuted to `(C, N, d)` before the matmul and back to
`(B, C, H, W)` after, exactly as the un-fused forward pass already does — see
`wbmm.forward` / `wbmm._forward_deployed` for the literal reshape/permute
calls either side of the matmul.)

Both targets fold BatchNorm, the shortcut, and (at S4) the two extra
depthwise branches into the *same* compact `(C,(2w-1)^2)` relative-position
table every WBMM block already had (renamed `R_fused`) plus one new small
`(C,)` `bias_fused` — nothing bigger ever replaces what fusion deletes, so
the **default target is always smaller** than the checkpoint you started
from. `--cache` takes that compact `R_fused` one step further and expands it
into a dense `(C,d,d)` `M_fused` (Sec. 3.4.5's WBMM-C matrix), removing the
per-call `index_select` entirely — but a dense `d^2`-per-channel matrix is
usually bigger than the compact `(2w-1)^2`-per-channel table it replaces (7x7
window: 169 -> 2401 per channel, ~14x), so **this target is usually larger**
than the checkpoint you started from, despite deleting the same BatchNorm.
Pick the default if you want the smallest possible checkpoint and don't mind
`index_select`; pick `--cache` if you want to remove `index_select` from
every forward call and can afford the larger checkpoint. See "What you get"
below for the numbers, and [`wbmm_reparam.py`](wbmm_reparam.py)'s
`wbmm.reparameterize_wbmm` docstring for the full derivation of why the
shortcut/BatchNorm/multi-kernel fold into the compact table just as cleanly
as into the dense one.

This is **lossless** either way: both fused models reproduce the original
model's output bit-for-bit up to floating point rounding (empirically ~1e-7
relative error in fp32 across full WBMM-P/N/T/S forward passes — see
[`tests/test_equivalence.py`](tests/test_equivalence.py)), and therefore
agree with *each other* too (`test_deploy_nocache_and_cache_agree`). It is
also *why* window size 7 and 14 aren't arbitrary: WBMM run with a `w x w`
window is *exactly* a depthwise conv of size `(2w-1) x (2w-1)` applied
independently inside each non-overlapping window (Theorem 3.1 + 3.2, Sec.
3.3) — `w=7 <=> 13x13` (UniRepLKNet's own "optimal" size) and `w=14 <=>
27x27`, the two large-kernel sizes the paper's own operator benchmarks
compare against. [`tests/test_equivalence.py`](tests/test_equivalence.py)
contains a runnable, from-scratch numerical proof of this
(`test_theorem_3_2_conv_equals_matmul`, `test_theorem_3_1_excess_kernel_is_inert`,
`test_window_equals_local_large_kernel`).

**Convert a trained checkpoint** (default target: compact, smaller file —
saved here with a `_deploy_nocache` suffix, matching the `fmt` string
`load_any_checkpoint` reports for it):
```bash
python reparameterize.py \
  --model wbmm_t --checkpoint wbmm_t_in1k_224_w7.pth \
  --output wbmm_t_in1k_224_w7_deploy_nocache.pth --check
```
...or the `--cache` target: no `index_select` left at inference, usually a
larger file (`_deploy_cache` suffix):
```bash
python reparameterize.py \
  --model wbmm_t --checkpoint wbmm_t_in1k_224_w7.pth \
  --output wbmm_t_in1k_224_w7_deploy_cache.pth --cache --check
```

**Evaluate the fused checkpoints** (real ImageNet-1K Top-1/Top-5, not just
the `--check` flag's random-input sanity check above): `main.py --eval`
cannot load either fused checkpoint directly, since it always builds the
un-fused architecture from [`wbmm.py`](wbmm.py) — use
[`eval_deploy.py`](eval_deploy.py) instead, which shares the exact same
dataset/transform (`datasets.py`) and evaluation loop (`engine.evaluate`), so
the numbers are directly comparable:
```bash
# fused, compact (deploy_nocache)
python eval_deploy.py --model wbmm_t \
  --resume wbmm_t_in1k_224_w7_deploy_nocache.pth --input_size 224 \
  --data_path /path/to/imagenet-1k

# fused, dense (deploy_cache)
python eval_deploy.py --model wbmm_t \
  --resume wbmm_t_in1k_224_w7_deploy_cache.pth --input_size 224 \
  --data_path /path/to/imagenet-1k

# 8 GPUs, either checkpoint
python -m torch.distributed.launch --nproc_per_node=8 eval_deploy.py \
  --model wbmm_t --resume wbmm_t_in1k_224_w7_deploy_nocache.pth \
  --input_size 224 --data_path /path/to/imagenet-1k
```
`eval_deploy.py` auto-detects the checkpoint format
(`probe_checkpoint_format`) and builds whichever architecture matches, so the
same command also accepts the *original* `wbmm_t_in1k_224_w7.pth` and
reproduces `main.py --eval true`'s Acc@1/Acc@5 exactly. All three checkpoints
(original, `deploy_nocache`, `deploy_cache`) should report the same accuracy
up to float rounding — that agreement is the real-data confirmation that
fusion is lossless, on top of the synthetic proof in
[`tests/test_equivalence.py`](tests/test_equivalence.py).



**What you get:** fusing collapses the S4 multi-kernel block (4 BatchNorms +
2 extra depthwise convs) into a single table/matrix lookup, and every other
WBMM block loses its BatchNorm the same way, so each fused block does
strictly less work than its un-fused counterpart — how much that moves
whole-network latency depends on your hardware and how much of it is spent
elsewhere (downsample/FFN layers, kernel-launch and BatchNorm overhead are
typically more prominent on GPU than on CPU). Storage moves in **opposite directions** depending on which
target you pick: the **default** folds everything directly into the
*existing* compact `(2w-1)^2`-per-channel table (`R_fused`, same shape as
before) plus one new small `(C,)` bias — nothing bigger ever replaces what
fusion deletes, so both nn.Parameter count *and* total stored-tensor count
strictly **decrease**, and the checkpoint file gets **smaller**. `--cache`
additionally expands that same table into a dense `(C,d,d)` `M_fused`
*buffer*, so while nn.Parameter count still drops slightly (BatchNorm affine
params still disappear), total stored-tensor count — and file size — usually
goes **up**: for a 7x7 window, `(2*7-1)^2=169` per channel becomes
`49*49=2401` per channel, roughly 14x larger for every WBMM block's matrix,
which dominates the BatchNorm savings. nn.Parameter count alone is a
misleading proxy for the resulting file size in *either* direction,
precisely because `R_fused` / `M_fused` are buffers, not `nn.Parameter`s —
the conversion script's own printout reports nn.Parameter count, total
stored-tensor count, and the actual before/after file size side by side,
with a target-specific explanation of which way and why, so this isn't easy
to miss. [`tests/test_equivalence.py`](tests/test_equivalence.py)'s
`test_nocache_fusion_strictly_shrinks_storage` and
`test_cache_fusion_grows_storage_for_window7` assert exactly this trade-off
numerically rather than only in prose.



## Citation

```bibtex
@article{wan2026wbmm,
  title         = {WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution},
  author        = {Song, Wan and Zhou, Wei and Wang, Rui and Yu, Jun and Kurihara, Toru and Xu, Jiajia and Zhan, Shu},
  journal       = {arXiv preprint arXiv:2607.02097},
  year          = {2026},
  eprint        = {2607.02097},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## Acknowledgement and License

Built on [UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet) (same training
pipelines, for a fair comparison). The detection / segmentation code further
depends on [MMDetection](https://github.com/open-mmlab/mmdetection) and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Released under
the **Apache 2.0 License** — see [LICENSE](LICENSE).
