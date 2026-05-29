# WBMM for Semantic Segmentation

This folder contains the WBMM implementation for semantic segmentation on
ADE20K.

The code is developed on top of
[MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0),
following the same setup as
[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet) so the comparison is
fair — only the backbone operator (WBMM) differs.

> **Released models:** WBMM-T and WBMM-S, both with the **7×7** window.

## Install

```bash
git clone https://github.com/wansong-s/WBMM.git
cd WBMM/segmentation

conda create -n wbmm_seg python=3.7 -y
conda activate wbmm_seg

# install torch matching your CUDA, e.g. torch 1.11 + CUDA 11.3:
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
pip install opencv-python termcolor yacs pyyaml scipy
```

WBMM needs **no** custom large-kernel CUDA extension.

## Data preparation

Prepare ADE20K following the
[MMSegmentation guide](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets).

## Evaluation

Download the checkpoints from
[Hugging Face](https://huggingface.co/wansong-s/WBMM). The backbone weights are
stored inside the full segmentor checkpoint, so we skip the separate ImageNet
init with `model.backbone.init_cfg.checkpoint=None`.

Single GPU, WBMM-T:

```bash
python test.py \
  configs/ade20k/upernet_wbmm_t_512_160k_ade20k.py \
  upernet_wbmm_t_ade20k_w7.pth \
  --eval mIoU \
  --cfg-options model.backbone.init_cfg.checkpoint=None
```

8 GPUs, WBMM-S:

```bash
sh dist_test.sh \
  configs/ade20k/upernet_wbmm_s_512_160k_ade20k.py \
  upernet_wbmm_s_ade20k_w7.pth \
  8 --eval mIoU \
  --cfg-options model.backbone.init_cfg.checkpoint=None
```

Multi-scale testing: uncomment `img_ratios` and set `flip=True` in the
`test_pipeline` of the config.

Expected results (UPerNet, 160k):

| backbone | mIoU (SS) | mIoU (MS) |
|:---------|:---------:|:---------:|
| WBMM-T (7×7) | 48.3 | 48.8 |
| WBMM-S (7×7) | 50.2 | 50.5 |

## Training

1. Make sure `init_cfg.checkpoint` in the config points to the downloaded
   **dense-task** ImageNet backbone (`wbmm_t_dense_in1k_224_w7.pth` /
   `wbmm_s_dense_in1k_224_w7.pth`).
2. Run:

```bash
sh dist_train.sh configs/ade20k/upernet_wbmm_t_512_160k_ade20k.py 8
```

(8 GPUs, total batch size 16 — identical schedule to UniRepLKNet-T.)

### Slurm

```bash
GPUS=32 sh slurm_train.sh <partition> <job-name> \
  configs/ade20k/upernet_wbmm_s_512_160k_ade20k.py
```

## Image demo

```bash
CUDA_VISIBLE_DEVICES=0 python image_demo.py \
  data/ade/ADEChallengeData2016/images/validation/ADE_val_00000591.jpg \
  configs/ade20k/upernet_wbmm_t_512_160k_ade20k.py \
  upernet_wbmm_t_ade20k_w7.pth \
  --palette ade20k
```

## Inference-time (deploy) structure

The backbone can be **constructed directly in its inference form** by passing
`model.backbone.deploy=True`, which removes the training-only structures
(BatchNorms and the bias term in GRN). A `reparameterize.py` helper is included
and will call `reparameterize_wbmm()` on the backbone to fold a *trained*
checkpoint into the deploy form — implement that method on the backbone if you
need lossless train→deploy conversion for your own checkpoints.
