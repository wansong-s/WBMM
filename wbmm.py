# --------------------------------------------------------
# WBMM: Windowed Batch Matrix Multiplication for Efficient
#       Large Receptive Field Convolution
# Paper : https://arxiv.org/abs/XXXX.XXXXX  (update with the arXiv id)
# Code  : https://github.com/wansong-s/WBMM
# Weights: https://huggingface.co/wansong-s/WBMM
# Licensed under the Apache 2.0 License [see LICENSE for details]
#
# This file is the IMAGE-CLASSIFICATION model. The training / evaluation code
# (main.py, engine.py, optim_factory.py, datasets.py, utils.py) is kept
# identical to UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet) on
# purpose: every training factor except the operator is held fixed, so the
# comparison isolates the effect of the proposed WBMM operator.
#
# Notation (matches the paper, Table 3):
#   'W' = a WBMM block   (windowed batch matrix multiplication)
#   'D' = a 3x3 depthwise-convolution block
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from functools import partial
import torch.utils.checkpoint as checkpoint

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None   # `pip install huggingface_hub` to auto-download weights

# ---------------------------------------------------------------------------
# Optional: register WBMM as an MMDetection / MMSegmentation backbone so this
# single file can be dropped into either framework. Keep ONE of the two blocks
# if you hit an import conflict.
# ---------------------------------------------------------------------------
has_mmseg = False
has_mmdet = False
try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    get_root_logger = None
    _load_checkpoint = None

# try:
#     from mmdet.models.builder import BACKBONES as det_BACKBONES
#     from mmdet.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmdet = True
# except ImportError:
#     get_root_logger = None
#     _load_checkpoint = None


class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization), proposed in ConvNeXt V2.
    Inputs are assumed to be (N, H, W, C). """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


def get_bn(dim, use_sync_bn=False):
    return nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


# ===========================================================================
#                         The WBMM operator
# ===========================================================================
class wbmm(nn.Module):
    """ Windowed Batch Matrix Multiplication (WBMM).

    Instead of *gathering* k*k scattered neighbours per output (as a depthwise
    convolution does), WBMM partitions the feature map into contiguous, non
    overlapping windows and builds a per-channel weight matrix
        M in R^{C x (wh*ww) x (wh*ww)}
    by indexing a compact relative-position-bias table
        R in R^{C x (2*wh-1)*(2*ww-1)}.
    The output is a single batched matrix multiply on contiguous memory, so the
    throughput *improves* with larger windows (opposite to depthwise conv).

    NOTE ON WINDOW SIZE
    -------------------
    `window_size` can be ANY (wh, ww); we use 7x7 by default because it matches
    the receptive field that the rest of the network is calibrated for and
    keeps the bias table tiny ((2*7-1)^2 = 169 per channel). A larger window
    simply grows the bias table accordingly. The code below is fully
    window-agnostic.

    `small_kernel` / `num_i` enable the *multi-kernel fusion* used only at the
    last stage (S4, num_i == 3) of the smallest variants (WBMM-P / WBMM-N),
    where the 7x7 feature map equals the window and two parallel depthwise
    paths (3x3 + 5x5) are added to the WBMM branch. At inference these parallel
    paths fuse into M, so they cost nothing.
    """
    def __init__(self, dim, small_kernel="False", num_i=0, window_size=(7, 7)):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.small_kernel = small_kernel
        self.num_i = num_i

        # multi-kernel fusion branch (only built for S4 of P/N variants)
        if small_kernel == "True" and num_i == 3:
            self.dwconv3 = nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False)
            self.dwconv5 = nn.Conv2d(dim, dim, 5, stride=1, padding=2, groups=dim, bias=False)
            self.bn1 = nn.BatchNorm2d(dim)   # BN on the WBMM main branch
            self.bn2 = nn.BatchNorm2d(dim)   # BN on the 5x5 path
            self.bn3 = nn.BatchNorm2d(dim)   # BN on the 3x3 path

        # compact relative-position-bias table  R in R^{C x (2wh-1)(2ww-1)}
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(dim, (2 * window_size[0] - 1) * (2 * window_size[1] - 1)))

        # precompute the relative-position index matrix  I in Z^{(wh*ww) x (wh*ww)}
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))           # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                            # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()      # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                    # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _build_matrix(self, C):
        # M = R[:, I.flatten()].view(C, d, d)
        return torch.index_select(
            self.relative_position_bias_table, 1, self.relative_position_index.view(-1)
        ).view(C, self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1])

    def forward(self, x):
        B, C, H, W = x.shape
        if H == self.window_size[0] and W == self.window_size[1]:
            # the whole feature map IS a single window (e.g. S4 at 7x7)
            if self.small_kernel == "True" and self.num_i == 3:
                # Y = WBMM(X) + BN(DW5(X)) + BN(DW3(X))  (paper Sec. 3.7)
                x2 = self.bn2(self.dwconv5(x)) + self.bn3(self.dwconv3(x))
                x = x.reshape(B, C, H * W).transpose(0, 1)
                x = x @ self._build_matrix(C) + x
                x = x.reshape(C, B, H, W).transpose(0, 1)
                x = self.bn1(x) + x2
            else:
                x = x.reshape(B, C, H * W).transpose(0, 1)
                x = x @ self._build_matrix(C) + x
                x = x.reshape(C, B, H, W).transpose(0, 1)
        else:
            # partition into (H/wh) x (W/ww) contiguous windows, then batched matmul.
            # (classification feature maps 56/28/14/7 are all divisible by 7; the
            #  segmentation / detection backbones additionally zero-pad for
            #  arbitrary input sizes -- see the downstream backbone files.)
            x = x.reshape(B, C, H // self.window_size[0], self.window_size[0],
                          W // self.window_size[1], self.window_size[1]) \
                 .permute(1, 0, 2, 4, 3, 5) \
                 .reshape(C, B * (H // self.window_size[0]) * (W // self.window_size[1]),
                          self.window_size[0] * self.window_size[1])
            x = x @ self._build_matrix(C) + x
            x = x.reshape(C, B, H // self.window_size[0], W // self.window_size[1],
                          self.window_size[0], self.window_size[1]) \
                 .permute(1, 0, 2, 4, 3, 5).reshape(B, C, H, W)
        return x


def _normalize_kernel(k):
    """Per-block operator tokens (paper notation, Table 3):
    'W' -> WBMM block;  'D' -> 3x3 depthwise conv.
    Any other odd int (e.g. 5) -> a plain depthwise conv of that size (ablations)."""
    return k


class WBMMBlock(nn.Module):
    def __init__(self,
                 dim, small_kernel, num_i,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4,
                 window_size=(7, 7)):
        super().__init__()
        self.with_cp = with_cp
        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        kernel_size = _normalize_kernel(kernel_size)
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size == 'W':
            # ---- WBMM block ; window can be any size, default 7x7 ----
            self.dwconv = wbmm(dim, small_kernel, num_i, window_size=window_size)
        elif kernel_size == 'D':
            # ---- 3x3 depthwise conv block ----
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                    dilation=1, groups=dim, bias=deploy)
        else:
            # any other (odd) depthwise kernel size, e.g. 5 -- handy for ablations
            assert isinstance(kernel_size, int) and kernel_size % 2 == 1
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=dim, bias=deploy)

        self.norm = nn.Identity() if (deploy or kernel_size == 0) else get_bn(dim, use_sync_bn=use_sync_bn)
        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(nn.Linear(ffn_dim, dim), NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) \
            if (not deploy) and layer_scale_init_value is not None and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):
        def _f(x):
            return x + self.compute_residual(x)
        if self.with_cp and inputs.requires_grad:
            return checkpoint.checkpoint(_f, inputs)
        return _f(inputs)


# ===========================================================================
#  Default per-stage block patterns (image-classification variants)
#  'W' = WBMM block, 'D' = 3x3 depthwise conv.   (paper Table 3, classif. row)
#    S1 = all 'D'  -> classification favours local 3x3 at the highest-res stage
#    S4 = all 'W'
#  The dense-prediction backbones differ ONLY at S1 (S1 = [W,D,W]); they live
#  in detection/.../wbmm.py and segmentation/.../wbmm.py.
# ===========================================================================
default_WBMM_P_kernel_sizes = (('D', 'D'),
                               ('W', 'D'),
                               ('W', 'D', 'W', 'D', 'W', 'D'),
                               ('W', 'W'))
default_WBMM_N_kernel_sizes = (('D', 'D'),
                               ('W', 'D'),
                               ('W', 'D', 'W', 'D', 'W', 'D', 'W', 'D'),
                               ('W', 'W'))
default_WBMM_T_kernel_sizes = (('D', 'D', 'D'),
                               ('W', 'D', 'W'),
                               ('W', 'D', 'W', 'D', 'W', 'D', 'W', 'D', 'W',
                                'D', 'W', 'D', 'W', 'D', 'W', 'D', 'W', 'D'),
                               ('W', 'W', 'W'))
default_WBMM_S_kernel_sizes = (('D', 'D', 'D'),
                               ('W', 'D', 'W'),
                               ('W', 'D', 'D', 'W', 'D', 'D', 'W', 'D', 'D', 'W',
                                'D', 'D', 'W', 'D', 'D', 'W', 'D', 'D', 'W', 'D',
                                'D', 'W', 'D', 'D', 'W', 'D', 'D'),
                               ('W', 'W', 'W'))

WBMM_P_depths = (2, 2, 6, 2)
WBMM_N_depths = (2, 2, 8, 2)
WBMM_T_depths = (3, 3, 18, 3)
WBMM_S_depths = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    WBMM_P_depths: default_WBMM_P_kernel_sizes,
    WBMM_N_depths: default_WBMM_N_kernel_sizes,
    WBMM_T_depths: default_WBMM_T_kernel_sizes,
    WBMM_S_depths: default_WBMM_S_kernel_sizes,
}


class WBMM(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=(3, 3, 27, 3),
                 dims=(96, 192, 384, 768),
                 small_kernel="False",
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 kernel_sizes=None,
                 window_size=(7, 7),      # <-- arbitrary window; default 7x7
                 deploy=False,
                 with_cp=False,
                 init_cfg=None,
                 use_sync_bn=False,
                 **kwargs):
        super().__init__()

        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                print('=========== use default kernel size ')
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError("no default kernel size settings for the given depths, please "
                                 "specify a per-block pattern, e.g. (('D','D'),('W','D'),"
                                 "('W','D','W','D','W','D'),('W','W'))")
        print(kernel_sizes)
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        self.with_cp = with_cp
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print('=========== drop path rates: ', dp_rates)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")))

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            main_stage = nn.Sequential(
                *[WBMMBlock(dim=dims[i], small_kernel=small_kernel, num_i=i,
                            kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value, deploy=deploy,
                            with_cp=with_cp, use_sync_bn=use_sync_bn, window_size=window_size)
                  for j in range(depths[i])])
            self.stages.append(main_stage)
            cur += depths[i]

        last_channels = dims[-1]
        self.for_pretrain = init_cfg is None
        self.for_downstream = not self.for_pretrain
        if self.for_downstream:
            assert num_classes is None

        if self.for_pretrain:
            self.init_cfg = None
            self.norm = nn.LayerNorm(last_channels, eps=1e-6)
            self.head = nn.Linear(last_channels, num_classes)
            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
            self.output_mode = 'logits'
        else:
            self.init_cfg = init_cfg
            self.init_weights()
            self.output_mode = 'features'
            norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
            for i_layer in range(4):
                self.add_module(f'norm{i_layer}', norm_layer(dims[i_layer]))

    def init_weights(self):
        def load_state_dict(module, state_dict, strict=False, logger=None):
            unexpected_keys = []
            own_state = module.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    unexpected_keys.append(name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, whose dimensions in the model are '
                        '{} and whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            missing_keys = set(own_state.keys()) - set(state_dict.keys())
            err_msg = []
            if unexpected_keys:
                err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
            if missing_keys:
                err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
            err_msg = '\n'.join(err_msg)
            if err_msg:
                if strict:
                    raise RuntimeError(err_msg)
                elif logger is not None:
                    logger.warn(err_msg)
                else:
                    print(err_msg)

        logger = get_root_logger()
        assert self.init_cfg is not None
        ckpt_path = self.init_cfg['checkpoint']
        if ckpt_path is None:
            print('================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization')
        else:
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            load_state_dict(self, _state_dict, strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.output_mode == 'logits':
            for stage_idx in range(4):
                x = self.downsample_layers[stage_idx](x)
                x = self.stages[stage_idx](x)
            x = self.norm(x.mean([-2, -1]))
            return self.head(x)
        elif self.output_mode == 'features':
            outs = []
            for stage_idx in range(4):
                x = self.downsample_layers[stage_idx](x)
                x = self.stages[stage_idx](x)
                outs.append(self.__getattr__(f'norm{stage_idx}')(x))
            return outs
        else:
            raise ValueError('Defined new output mode?')


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


#   For easy use as a backbone in MMSegmentation. Ignore if you do not use MMSeg.
if has_mmseg:
    @seg_BACKBONES.register_module()
    class WBMMBackbone(WBMM):
        def __init__(self, depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), small_kernel="False",
                     drop_path_rate=0., layer_scale_init_value=1e-6, kernel_sizes=None,
                     window_size=(7, 7), deploy=False, with_cp=False, init_cfg=None):
            assert init_cfg is not None
            super().__init__(in_chans=3, num_classes=None, depths=depths, dims=dims, small_kernel=small_kernel,
                             drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                             kernel_sizes=kernel_sizes, window_size=window_size, deploy=deploy,
                             with_cp=with_cp, init_cfg=init_cfg, use_sync_bn=True)


# ===========================================================================
#                     HuggingFace pretrained weights
#  Upload your checkpoints to https://huggingface.co/wansong-s/WBMM using
#  EXACTLY these file names so that `pretrained=True` works out of the box.
# ===========================================================================
HF_REPO_ID = 'wansong-s/WBMM'
huggingface_file_names = {
    # ---- pure image-classification models (S1 = [D,D,D]; report Top-1) ----
    "wbmm_p_1k": "wbmm_p_in1k_224_w7.pth",
    "wbmm_n_1k": "wbmm_n_in1k_224_w7.pth",
    "wbmm_t_1k": "wbmm_t_in1k_224_w7.pth",
    "wbmm_s_1k": "wbmm_s_in1k_224_w7.pth",
    # ---- dense-task classification backbones (S1 = [W,D,W]) ----
    #      released for downstream det/seg; we do NOT report their Top-1.
    "wbmm_t_dense_1k": "wbmm_t_dense_in1k_224_w7.pth",
    "wbmm_s_dense_1k": "wbmm_s_dense_in1k_224_w7.pth",
}


def load_with_key(model, key):
    if hf_hub_download is not None:
        cache_file = hf_hub_download(repo_id=HF_REPO_ID, filename=huggingface_file_names[key])
        checkpoint = torch.load(cache_file, map_location='cpu')
    else:
        raise RuntimeError("Please `pip install huggingface_hub` to auto-download, "
                           "or pass a local checkpoint to --finetune / --resume.")
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)


def initialize_with_pretrained(model, model_name, in_1k_pretrained):
    if in_1k_pretrained:
        load_with_key(model, model_name + '_1k')


# ===========================================================================
#  Model factories (image classification).
#  P / N use the S4 multi-kernel fusion (small_kernel="True").
# ===========================================================================
@register_model
def wbmm_p(in_1k_pretrained=False, **kwargs):
    model = WBMM(depths=WBMM_P_depths, dims=(64, 128, 256, 512), small_kernel="True", **kwargs)
    initialize_with_pretrained(model, 'wbmm_p', in_1k_pretrained)
    return model


@register_model
def wbmm_n(in_1k_pretrained=False, **kwargs):
    model = WBMM(depths=WBMM_N_depths, dims=(80, 160, 320, 640), small_kernel="True", **kwargs)
    initialize_with_pretrained(model, 'wbmm_n', in_1k_pretrained)
    return model


@register_model
def wbmm_t(in_1k_pretrained=False, **kwargs):
    model = WBMM(depths=WBMM_T_depths, dims=(80, 160, 320, 640), small_kernel="False", **kwargs)
    initialize_with_pretrained(model, 'wbmm_t', in_1k_pretrained)
    return model


@register_model
def wbmm_s(in_1k_pretrained=False, **kwargs):
    model = WBMM(depths=WBMM_S_depths, dims=(96, 192, 384, 768), small_kernel="False", **kwargs)
    initialize_with_pretrained(model, 'wbmm_s', in_1k_pretrained)
    return model


# ----- dense-prediction classification backbones (S1 = [W,D,W]) -----
# Same width/depth as wbmm_t / wbmm_s but the first stage mixes W and D.
# These are the ImageNet checkpoints that the detection / segmentation
# configs initialise from. We deliberately do not advertise their Top-1.
dense_WBMM_T_kernel_sizes = (('W', 'D', 'W'),) + default_WBMM_T_kernel_sizes[1:]
dense_WBMM_S_kernel_sizes = (('W', 'D', 'W'),) + default_WBMM_S_kernel_sizes[1:]


@register_model
def wbmm_t_dense(in_1k_pretrained=False, **kwargs):
    model = WBMM(depths=WBMM_T_depths, dims=(80, 160, 320, 640), small_kernel="False",
                 kernel_sizes=dense_WBMM_T_kernel_sizes, **kwargs)
    initialize_with_pretrained(model, 'wbmm_t_dense', in_1k_pretrained)
    return model


@register_model
def wbmm_s_dense(in_1k_pretrained=False, **kwargs):
    model = WBMM(depths=WBMM_S_depths, dims=(96, 192, 384, 768), small_kernel="False",
                 kernel_sizes=dense_WBMM_S_kernel_sizes, **kwargs)
    initialize_with_pretrained(model, 'wbmm_s_dense', in_1k_pretrained)
    return model


if __name__ == '__main__':
    # quick smoke test. The default 7x7 window runs at any resolution divisible
    # by 7 (e.g. 224 -> stages 56/28/14/7). `window_size` is configurable; just
    # make sure the last stage stays >= the window for the pure-classification op.
    for name in ['wbmm_p', 'wbmm_n', 'wbmm_t', 'wbmm_s']:
        m = globals()[name]().eval()
        y = m(torch.randn(2, 3, 224, 224))
        print(f'{name}: input=224 -> logits={tuple(y.shape)} '
              f'params={sum(p.numel() for p in m.parameters())/1e6:.2f}M')
