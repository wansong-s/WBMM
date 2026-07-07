#!/usr/bin/env python3
# --------------------------------------------------------
# WBMM: Windowed Batch Matrix Multiplication
# Code  : https://github.com/wansong-s/WBMM
# Licensed under the Apache 2.0 License [see LICENSE for details]
#
# eval_deploy.py  (image classification)
# =======================================
# Real ImageNet-1K Top-1 / Top-5 accuracy for the checkpoints produced by
# `reparameterize.py` -- `deploy_nocache` (compact, R_fused) and
# `deploy_cache` (dense, M_fused) -- using the *exact same* dataset/
# transform (datasets.py) and evaluation loop (engine.evaluate) that
# `main.py --eval true` uses, so the numbers are directly comparable.
#
# Why this script exists: `main.py` only ever imports the un-fused
# architecture (`from wbmm import *`) and calls a strict
# `model.load_state_dict()`, so it cannot load a fused checkpoint (different
# keys -- `R_fused`/`bias_fused` or `M_fused`/`bias_fused` instead of
# `relative_position_bias_table` + BatchNorm running stats). This script
# builds the model via `reparameterize.load_any_checkpoint`, which inspects
# the checkpoint's own keys (`probe_checkpoint_format`) and constructs
# whichever of the three architectures matches -- so the very same command
# also evaluates a PLAIN (un-fused) checkpoint, reproducing `main.py --eval`'s
# numbers exactly (see tests/test_equivalence.py for the numerical proof
# that all three formats agree with each other on every forward pass).
#
# Typical workflow -- one run per checkpoint you want metrics for:
#
#   # 1) original, un-fused checkpoint (same numbers as `main.py --eval true`)
#   python eval_deploy.py --model wbmm_t \
#       --resume wbmm_t_in1k_224_w7.pth \
#       --input_size 224 --data_path /path/to/imagenet-1k
#
#   # 2) fused, compact -- reparameterize.py's default target
#   python reparameterize.py --model wbmm_t \
#       --checkpoint wbmm_t_in1k_224_w7.pth \
#       --output wbmm_t_in1k_224_w7_deploy_nocache.pth --check
#   python eval_deploy.py --model wbmm_t \
#       --resume wbmm_t_in1k_224_w7_deploy_nocache.pth \
#       --input_size 224 --data_path /path/to/imagenet-1k
#
#   # 3) fused, dense -- reparameterize.py --cache target
#   python reparameterize.py --model wbmm_t \
#       --checkpoint wbmm_t_in1k_224_w7.pth \
#       --output wbmm_t_in1k_224_w7_deploy_cache.pth --cache --check
#   python eval_deploy.py --model wbmm_t \
#       --resume wbmm_t_in1k_224_w7_deploy_cache.pth \
#       --input_size 224 --data_path /path/to/imagenet-1k
#
# All three commands above should report the same Acc@1 / Acc@5 (up to
# float rounding, typically identical to 3 decimal places) -- that agreement
# *is* the point: it is the real-data confirmation that fusion is lossless,
# on top of the synthetic proof in tests/test_equivalence.py.
#
# 8-GPU distributed evaluation works exactly like main.py's own commands:
#   python -m torch.distributed.launch --nproc_per_node=8 eval_deploy.py \
#       --model wbmm_t --resume wbmm_t_in1k_224_w7_deploy_nocache.pth \
#       --input_size 224 --data_path /path/to/imagenet-1k
# --------------------------------------------------------
import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datasets import build_dataset
from engine import evaluate
import utils
from reparameterize import load_any_checkpoint


def str2bool(v):
    """Converts string to bool type; enables command line arguments in the
    format of '--arg1 true --arg2 false' (matches main.py's own helper)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser(
        'WBMM checkpoint evaluation (plain / deploy_nocache / deploy_cache)', add_help=False)

    # Model
    parser.add_argument('--model', default='wbmm_t', type=str, metavar='MODEL',
                        help='model name registered with timm, e.g. wbmm_p / wbmm_n / wbmm_t / wbmm_s '
                             '/ wbmm_t_dense / wbmm_s_dense')
    parser.add_argument('--resume', required=True, type=str,
                        help='checkpoint to evaluate: a PLAIN (un-fused) checkpoint, a `_deploy_nocache` '
                             '(R_fused) checkpoint, or a `_deploy_cache` (M_fused) checkpoint -- the format '
                             'is auto-detected from the checkpoint\'s own keys '
                             '(reparameterize.probe_checkpoint_format), so this one flag covers all three')
    parser.add_argument('--window_size', default=7, type=int,
                        help='window size the checkpoint was trained/fused with (square window; default '
                             '7, the setting used by every released checkpoint)')
    parser.add_argument('--input_size', default=224, type=int, help='image input size')
    parser.add_argument('--crop_pct', type=float, default=None)

    # Dataset (same options/semantics as main.py)
    parser.add_argument('--data_path', default='/path/to/imagenet-1k', type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path when --data_set image_folder (ignored otherwise)')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'], type=str)
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of classes; must match --resume (overwritten by the actual count '
                             'for CIFAR/IMNET, asserted against for image_folder)')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)

    # Evaluation loop
    parser.add_argument('--batch_size', default=64, type=int,
                        help='per-GPU batch size (the eval loader uses 1.5x this, matching main.py)')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)
    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    parser.add_argument('--dist_eval', type=str2bool, default=True, help='enable distributed evaluation')

    parser.add_argument('--device', default='cuda', help='device to use for evaluation')
    parser.add_argument('--seed', default=0, type=int)

    # Distributed (unused on a single GPU; matches main.py's flags so the
    # same launch commands work unmodified)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process '
                  'number. This will slightly alter validation results as extra duplicate entries are '
                  'added to achieve equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)

    # Auto-detects plain / deploy_nocache / deploy_cache from --resume's own
    # keys and builds the matching architecture (see reparameterize.py).
    model, fmt = load_any_checkpoint(
        args.model, args.resume,
        num_classes=args.nb_classes,
        window_size=(args.window_size, args.window_size))
    print(f"Loaded '{args.resume}'  -- detected checkpoint format = '{fmt}' "
          f"('plain' = un-fused, 'deploy_nocache' = fused/compact R_fused, "
          f"'deploy_cache' = fused/dense M_fused)")
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                           find_unused_parameters=False)

    start_time = time.time()
    test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
    eval_time = time.time() - start_time

    print(f"[{fmt}] Accuracy of the network on {len(dataset_val)} test images: "
          f"Acc@1 {test_stats['acc1']:.5f}%  Acc@5 {test_stats['acc5']:.5f}%  "
          f"loss {test_stats['loss']:.5f}  (eval took {eval_time:.1f}s)")
    return test_stats, fmt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'WBMM checkpoint evaluation (plain / deploy_nocache / deploy_cache)', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
