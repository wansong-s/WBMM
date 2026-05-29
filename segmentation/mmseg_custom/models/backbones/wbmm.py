# --------------------------------------------------------
# WBMM: Windowed Batch Matrix Multiplication
# Code  : https://github.com/wansong-s/WBMM
# Weights: https://huggingface.co/wansong-s/WBMM
# Licensed under the Apache 2.0 License [see LICENSE for details]
#
# WBMM backbone for MMSegmentation (UPerNet, etc.).
#
# Difference from the pure image-classification model (../../wbmm.py):
#   * the WBMM operator zero-pads so it accepts ARBITRARY input sizes
#     (dense-prediction feature maps are not multiples of the window);
#   * the first stage S1 mixes operators -> [W, D, W] (paper Table 3,
#     segmentation row), which is what gives the higher downstream mIoU at the
#     cost of a slightly lower ImageNet Top-1 (so we do not report the Top-1
#     of these backbones).
#
# Notation:  'W' = WBMM block,  'D' = 3x3 depthwise conv.
# The window can be any size; it defaults to 7x7.
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from functools import partial
import torch.utils.checkpoint as checkpoint

has_mmdet = False
has_mmseg = False
try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    get_root_logger = None
    _load_checkpoint = None


class GRNwithNHWC(nn.Module):
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


class wbmm(nn.Module):
    """ WBMM operator, padding-safe variant for dense prediction.

    Identical maths to the classification operator, but it zero-pads the input
    so the spatial dims become divisible by the window, runs the batched matrix
    multiplication, then crops back. This lets the same window (default 7x7)
    process feature maps of arbitrary size. `window_size` may be any (wh, ww).
    """
    def __init__(self, dim, window_size=(7, 7)):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(dim, (2 * window_size[0] - 1) * (2 * window_size[1] - 1)))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        mod_pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='constant', value=0)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.window_size[0], self.window_size[0],
                      W // self.window_size[1], self.window_size[1]) \
             .permute(1, 0, 2, 4, 3, 5) \
             .reshape(C, B * (H // self.window_size[0]) * (W // self.window_size[1]),
                      self.window_size[0] * self.window_size[1])
        matrix = torch.index_select(self.relative_position_bias_table, 1,
                                    self.relative_position_index.view(-1)).view(
            C, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])
        x = x @ matrix + x
        x = x.reshape(C, B, H // self.window_size[0], W // self.window_size[1],
                      self.window_size[0], self.window_size[1]) \
             .permute(1, 0, 2, 4, 3, 5).reshape(B, C, H, W)
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = x[:, :, :h, :w]
        return x


def _normalize_kernel(k):
    # 'W' = WBMM block, 'D' = 3x3 depthwise conv; any other odd int = plain DW conv.
    return k


class WBMMBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_path=0., layer_scale_init_value=1e-6,
                 deploy=False, with_cp=False, use_sync_bn=False, ffn_factor=4, window_size=(7, 7)):
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
            self.dwconv = wbmm(dim, window_size=window_size)   # window: any size, default 7x7
        elif kernel_size == 'D':
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                    dilation=1, groups=dim, bias=deploy)
        else:
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
            self.pwconv2 = nn.Sequential(nn.Linear(ffn_dim, dim, bias=False), NHWCtoNCHW(),
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


# Dense-prediction block patterns: S1 = [W, D, W]  (paper Table 3, seg. row)
default_WBMM_T_kernel_sizes = (('W', 'D', 'W'),
                               ('W', 'D', 'W'),
                               ('W', 'D', 'W', 'D', 'W', 'D', 'W', 'D', 'W',
                                'D', 'W', 'D', 'W', 'D', 'W', 'D', 'W', 'D'),
                               ('W', 'W', 'W'))
default_WBMM_S_kernel_sizes = (('W', 'D', 'W'),
                               ('W', 'D', 'W'),
                               ('W', 'D', 'D', 'W', 'D', 'D', 'W', 'D', 'D', 'W',
                                'D', 'D', 'W', 'D', 'D', 'W', 'D', 'D', 'W', 'D',
                                'D', 'W', 'D', 'D', 'W', 'D', 'D'),
                               ('W', 'W', 'W'))

WBMM_T_depths = (3, 3, 18, 3)
WBMM_S_depths = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    WBMM_T_depths: default_WBMM_T_kernel_sizes,
    WBMM_S_depths: default_WBMM_S_kernel_sizes,
}


class WBMM(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=(3, 3, 27, 3), dims=(96, 192, 384, 768),
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., kernel_sizes=None,
                 window_size=(7, 7), deploy=False, with_cp=False, init_cfg=None, use_sync_bn=False, **kwargs):
        super().__init__()
        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                print('=========== use default kernel size ')
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError("no default kernel size settings for the given depths, please "
                                 "specify a per-block pattern, e.g. (('W','D','W'),('W','D','W'),"
                                 "('W','D',...),('W','W','W'))")
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
                *[WBMMBlock(dim=dims[i], kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
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


if has_mmseg:
    @seg_BACKBONES.register_module()
    class WBMMBackbone(WBMM):
        def __init__(self, depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), drop_path_rate=0.,
                     layer_scale_init_value=1e-6, kernel_sizes=None, window_size=(7, 7),
                     deploy=False, with_cp=False, init_cfg=None):
            assert init_cfg is not None
            super().__init__(in_chans=3, num_classes=None, depths=depths, dims=dims,
                             drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                             kernel_sizes=kernel_sizes, window_size=window_size, deploy=deploy,
                             with_cp=with_cp, init_cfg=init_cfg, use_sync_bn=True)


@register_model
def wbmm_t(**kwargs):
    return WBMM(depths=WBMM_T_depths, dims=(80, 160, 320, 640), **kwargs)


@register_model
def wbmm_s(**kwargs):
    return WBMM(depths=WBMM_S_depths, dims=(96, 192, 384, 768), **kwargs)
