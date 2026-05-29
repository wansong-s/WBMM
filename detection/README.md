# WBMM for Object Detection

This folder contains the WBMM implementation for object detection on COCO.

The code is developed on top of
[MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/tree/v2.28.1),
following the same setup as
[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet) so the comparison is
fair — only the backbone operator (WBMM) differs.

> **Released models:** WBMM-T and WBMM-S, both with the **7×7** window.

## Install

```bash
git clone https://github.com/wansong-s/WBMM.git
cd WBMM/detection

conda create -n wbmm_det python=3.7 -y
conda activate wbmm_det

# install torch matching your CUDA, e.g. torch 1.11 + CUDA 11.3:
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
pip install opencv-python termcolor yacs pyyaml scipy
```

WBMM needs **no** custom large-kernel CUDA extension.

## Data preparation

Prepare COCO following the
[MMDetection guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

## Evaluation

Download the checkpoints from
[Hugging Face](https://huggingface.co/wansong-s/WBMM). The backbone weights are
stored inside the full detector checkpoint, so we skip the separate ImageNet
init with `model.backbone.init_cfg.checkpoint=None`.

Single GPU, WBMM-T:

```bash
python test.py \
  configs/coco/casc_mask_rcnn_wbmm_t_in1k_fpn_3x_coco.py \
  casc_mask_rcnn_wbmm_t_coco_w7.pth \
  --eval bbox segm \
  --cfg-options model.backbone.init_cfg.checkpoint=None
```

8 GPUs, WBMM-S:

```bash
sh dist_test.sh \
  configs/coco/casc_mask_rcnn_wbmm_s_in1k_fpn_3x_coco.py \
  casc_mask_rcnn_wbmm_s_coco_w7.pth \
  8 --eval bbox segm \
  --cfg-options model.backbone.init_cfg.checkpoint=None
```

Expected results (Cascade Mask R-CNN, 3×):

| backbone | AP<sup>box</sup> | AP<sup>mask</sup> |
|:---------|:----:|:----:|
| WBMM-T (7×7) | 51.6 | 44.8 |
| WBMM-S (7×7) | 52.8 | 45.6 |

## Training on COCO

1. Make sure `pretrained` in the config points to the downloaded **dense-task**
   ImageNet backbone (`wbmm_t_dense_in1k_224_w7.pth` / `wbmm_s_dense_in1k_224_w7.pth`).
2. Run:

```bash
sh dist_train.sh configs/coco/casc_mask_rcnn_wbmm_t_in1k_fpn_3x_coco.py 8
```

(8 GPUs, total batch size 16 — identical schedule to UniRepLKNet-T.)

### Slurm

```bash
GPUS=32 sh slurm_train.sh <partition> <job-name> \
  configs/coco/casc_mask_rcnn_wbmm_s_in1k_fpn_3x_coco.py your_work_dir
```

## Inference-time (deploy) structure

The backbone can be **constructed directly in its inference form** by passing
`model.backbone.deploy=True`, which removes the training-only structures
(BatchNorms and the bias term in GRN). A `reparameterize.py` helper is included
and will call `reparameterize_wbmm()` on the backbone to fold a *trained*
checkpoint into the deploy form — implement that method on the backbone if you
need lossless train→deploy conversion for your own checkpoints.
