<h1 align="center">WBMM</h1>

<p align="center">
  <b>Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution</b>
</p>

<p align="center">
  <a href="https://github.com/wansong-s/WBMM">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="Code">
  </a>
  &nbsp;
  <a href="https://huggingface.co/wansong-s/WBMM">
    <img src="https://img.shields.io/badge/Weights-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=white" alt="Weights">
  </a>
  &nbsp;
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-4c8eda?style=flat-square" alt="License">
  </a>
</p>

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
| WBMM-T (dense) | 7×7 | [wbmm_t_dense_in1k_224_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_t_dense_in1k_224_w7.pth) |
| WBMM-S (dense) | 7×7 | [wbmm_s_dense_in1k_224_w7.pth](https://huggingface.co/wansong-s/WBMM/resolve/main/wbmm_s_dense_in1k_224_w7.pth) |

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
  --batch_size 128 --lr 4e-3 --update_freq 4 \
  --mixup 0.3 --cutmix 0.3 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

**WBMM-N**
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wbmm_n --drop_path 0.1 \
  --batch_size 128 --lr 4e-3 --update_freq 4 \
  --mixup 0.5 --cutmix 0.5 \
  --data_path /path/to/imagenet-1k \
  --output_dir /path/to/save_results
```

**WBMM-T**
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --model wbmm_t --drop_path 0.2 \
  --batch_size 128 --lr 4e-3 --update_freq 4 \
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


## Citation

```bibtex
@article{wbmm,
  title  = {WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution},
  author = {Wang, Song and others},
  year   = {2025}
}
```

## Acknowledgement and License

Built on [UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet) (same training
pipelines, for a fair comparison). The detection / segmentation code further
depends on [MMDetection](https://github.com/open-mmlab/mmdetection) and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Released under
the **Apache 2.0 License** — see [LICENSE](LICENSE).
