# --------------------------------------------------------
# WBMM: Windowed Batch Matrix Multiplication for Efficient
#       Large Receptive Field Convolution
# Paper : https://arxiv.org/abs/2607.02097  (ICML 2026, Spotlight)
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
#
# ----------------------------------------------------------------------------
# Relationship to wbmm.py
# ----------------------------------------------------------------------------
# This file is `wbmm.py` PLUS inference-time reparameterization support
# (BatchNorm / shortcut / multi-kernel fusion via `reparameterize_wbmm()`, and
# the WBMM-C dense-weight-cache hook `enable_cache()` / `disable_cache()`) --
# every nn.Parameter and buffer the ordinary training path (`main.py`) creates
# has the exact same name and shape here as in `wbmm.py` (see
# `tests/test_equivalence.py` and `reparameterize.py`'s own module docstring
# for the numerical proof), so any checkpoint `main.py` produces loads into
# either file interchangeably -- nothing about training changes.
#
# Only `reparameterize.py` and `tests/test_equivalence.py` import this file
# (as `wbmm_reparam`); `main.py` keeps importing plain `wbmm.py` and knows
# nothing about this one. Both files register the same timm model names
# (`wbmm_p` / `wbmm_n` / `wbmm_t` / `wbmm_s` / `*_dense`), so don't import both
# in the same process -- whichever is imported second wins the name in timm's
# global registry (the same caveat `WBMMBackbone`'s own `force=True` below
# deals with for mmseg's registry).
# ----------------------------------------------------------------------------
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


# ===========================================================================
#  Reparameterization / inference-time fusion helpers
#  ---------------------------------------------------------------------
#  These implement the *exact* algebraic fusions described in the paper:
#    - Sec. 3.3 (Theorem 3.2): depthwise conv  <=>  position-dependent matmul.
#      `_depthwise_kernel_to_bias_table` is this theorem specialised to one
#      non-overlapping window, expressed directly in the COMPACT
#      (C, (2wh-1)(2ww-1)) table representation that `wbmm`'s own
#      `relative_position_bias_table` (R) uses -- not the dense (C,d,d) form
#      -- since a depthwise kernel's weight, like R's, depends only on the
#      relative offset between input and output position. Indexing the
#      result with the SAME `I` every `wbmm` block already uses recovers the
#      dense form exactly (see tests/test_equivalence.py for a from-scratch
#      numerical proof and the un-windowed, fully general construction).
#    - Sec. 3.7: multi-kernel fusion, (WBMM+id)*BN1 + DW5*BN2 + DW3*BN3, used
#      only at S4 of WBMM-P / WBMM-N.
#    - Standard Conv/Linear + BatchNorm folding, used for the *outer*
#      `WBMMBlock.norm` that wraps every `dwconv` (whether it is a `wbmm`
#      operator or a plain depthwise Conv2d) and for the `pwconv2` FFN
#      projection.
#  Every fusion here is *lossless*: applied to an eval()-mode trained model,
#  it reproduces the training-time forward pass bit-for-bit up to floating
#  point round-off. See tests/test_equivalence.py for the numerical proof.
#
#  Two inference targets (Sec. 3.4.5's WBMM-NC / WBMM-C split applied to the
#  *fused* model, not just the un-fused one -- see `wbmm.reparameterize_wbmm`
#  below for the full explanation):
#    - `cache=False` (default): BatchNorm / shortcut / multi-kernel fold
#      straight into the existing COMPACT table (-> `R_fused`, same shape as
#      R), so a forward call still does one `index_select` -- exactly like
#      un-fused WBMM-NC -- but against a smaller, BatchNorm-free model:
#      nothing grows, so this *strictly reduces* stored parameters relative
#      to the un-fused checkpoint.
#    - `cache=True`: `R_fused` is additionally expanded, once, into a dense
#      (C,d,d) `M_fused`, removing the `index_select` entirely at the cost of
#      the compact-vs-dense blow-up described in the README (for a 7x7
#      window, 169 -> 2401 per channel) -- usually a net size *increase*.
# ===========================================================================

def _bn_scale_shift(bn):
    """Fold a BatchNorm2d / SyncBatchNorm (eval mode, i.e. using its
    *running* statistics) into an equivalent per-channel affine map
        BN(x) = x * scale + shift
    """
    var, eps = bn.running_var, bn.eps
    gamma = bn.weight if bn.weight is not None else torch.ones_like(var)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(var)
    mean = bn.running_mean
    std = torch.sqrt(var + eps)
    scale = gamma / std
    shift = beta - mean * scale
    return scale.detach().clone(), shift.detach().clone()


def _fuse_conv_bn(conv, bn):
    """Conv2d (any `groups`, including depthwise) + BatchNorm2d -> Conv2d."""
    scale, shift = _bn_scale_shift(bn)
    w = conv.weight.detach().clone() * scale.view(-1, 1, 1, 1)
    b = (conv.bias.detach().clone() if conv.bias is not None
         else torch.zeros_like(scale))
    b = b * scale + shift
    return w, b


def _fuse_linear_bn(linear, bn):
    """Linear (NHWC) + BatchNorm2d (applied after an NHWC->NCHW permute)
    -> Linear.  Valid because BN's per-channel affine map commutes with the
    permute: it always acts on the same feature/channel axis."""
    scale, shift = _bn_scale_shift(bn)
    w = linear.weight.detach().clone() * scale.view(-1, 1)
    b = (linear.bias.detach().clone() if linear.bias is not None
         else torch.zeros_like(scale))
    b = b * scale + shift
    return w, b


def _depthwise_kernel_to_bias_table(kernel, window_size):
    """Embed a literal (C,1,K,K) depthwise kernel (K odd, K <= 2*min(window)-1)
    into a COMPACT relative-position bias table of shape
    (C, (2wh-1)*(2ww-1)) -- the exact same shape and (offset -> flat index)
    convention as `wbmm`'s own `relative_position_bias_table` (R; see
    `_get_relative_position_index`, Eq. 6-7).

    A depthwise kernel's weight depends only on the relative offset
    (dh, dw) = (h_input - h_output, w_input - w_output) between input and
    output position -- exactly what R is indexed by -- so it embeds directly
    into R's compact table with NO expansion to a dense (C,d,d) matrix:
    entries outside the kernel's own (2*kh+1)x(2*kw+1) support are simply
    left at 0, contributing nothing (matching a literal depthwise conv,
    which also contributes nothing beyond its own support).

    Indexing the result the same way every `wbmm` block already does,
        table[:, I.flatten()].view(C, wh*ww, wh*ww)
    reproduces Theorem 3.2's construction specialised to one window (the
    dense form this function's predecessor,
    `_depthwise_kernel_to_window_matrix`, used to build directly) exactly --
    see tests/test_equivalence.py for the numerical proof both ways.
    """
    C, _, K, K2 = kernel.shape
    assert K == K2 and K % 2 == 1, "kernel must be square with an odd side"
    kh = kw = K // 2
    wh, ww = to_2tuple(window_size)
    assert K <= 2 * min(wh, ww) - 1, (
        f"a {K}x{K} kernel cannot be represented inside a {wh}x{ww} window "
        f"(max representable kernel is {2 * min(wh, ww) - 1}x{2 * min(wh, ww) - 1}, "
        f"c.f. Theorem 3.1)")
    table = kernel.new_zeros(C, 2 * wh - 1, 2 * ww - 1)
    r0, c0 = (wh - 1) - kh, (ww - 1) - kw   # (2wh-1, 2ww-1) grid is centred at (wh-1, ww-1) == offset (0,0)
    table[:, r0:r0 + K, c0:c0 + K] = kernel.reshape(C, K, K)
    return table.reshape(C, -1)


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
        x = torch.sigmoid(x)   # was F.sigmoid(x), deprecated since PyTorch 1.x; identical result
        return inputs * x.view(-1, self.input_channels, 1, 1)


# ===========================================================================
#  Shared relative-position-index cache
#  ---------------------------------------------------------------------
#  I in Z^{(wh*ww) x (wh*ww)} (Eq. 6-7) depends only on the window geometry
#  (wh, ww) -- never on the channel count C or on any learnable parameter --
#  so every WBMM block that shares a window size (e.g. every 7x7 block in
#  the whole network) shares an *identical* I. Instead of recomputing the
#  same meshgrid/broadcast in every single block's __init__, we build it
#  once per unique window size and hand back the cached tensor to every
#  later block that asks for the same size: computed once, reused by all.
# ===========================================================================
_REL_POS_INDEX_CACHE = {}


def _get_relative_position_index(window_size):
    window_size = to_2tuple(window_size)
    if window_size not in _REL_POS_INDEX_CACHE:
        wh, ww = window_size
        coords = torch.stack(torch.meshgrid(
            torch.arange(wh), torch.arange(ww), indexing='ij'))          # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                        # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += wh - 1
        relative_coords[:, :, 1] += ww - 1
        relative_coords[:, :, 0] *= 2 * ww - 1
        _REL_POS_INDEX_CACHE[window_size] = relative_coords.sum(-1).contiguous()
    return _REL_POS_INDEX_CACHE[window_size]


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

        # relative-position index matrix  I in Z^{(wh*ww) x (wh*ww)} -- shared
        # process-wide across every block with this window size, see
        # `_get_relative_position_index` above.
        self.register_buffer("relative_position_index", _get_relative_position_index(window_size))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.deployed = False   # flips to True after reparameterize_wbmm() (hard BN/shortcut fusion)

        # ---- WBMM-C / WBMM-NC inference cache (paper Sec. 3.4.5, Algorithm 1) ----
        # `use_cache=False` (default, "WBMM-NC"): M = R[:, I] (shape C,d,d) is
        #   rebuilt from R (shape C,(2wh-1)(2ww-1)) on every forward call --
        #   required during training, since gradients must flow through R,
        #   and the behaviour of every checkpoint/version prior to this one.
        # `use_cache=True` ("WBMM-C", turned on via `enable_cache()`): M is
        #   built once from the *current* table (R pre-deploy, or R_fused
        #   post-`reparameterize_wbmm(cache=False)`) and cached as a (C,d,d)
        #   buffer; every later forward call reuses it directly, skipping the
        #   `index_select` gather entirely -- just one batched matmul on the
        #   window-reshaped input. Inference-only: the table must stay frozen
        #   while the cache is active (entering .train() drops it
        #   automatically). This is a lighter-weight, fully reversible
        #   alternative to `reparameterize_wbmm(cache=True)` below, which
        #   additionally deletes the compact table outright, permanently
        #   trading it for a dense M_fused; once a block has gone through
        #   THAT, this toggle is a no-op (nothing left to cache from).
        self.use_cache = False
        self.register_buffer("M_cache", None, persistent=False)

    def _current_table(self):
        """The compact (C, (2wh-1)(2ww-1)) table this call should index into:
        `R_fused` once `reparameterize_wbmm(cache=False)` has run, otherwise
        the raw learnable `relative_position_bias_table`. Only meaningful
        when no dense `M_fused` exists yet -- callers check that first."""
        return self.R_fused if hasattr(self, 'R_fused') else self.relative_position_bias_table

    def _build_matrix(self, C):
        # M = table[:, I.flatten()].view(C, d, d)
        return torch.index_select(
            self._current_table(), 1, self.relative_position_index.view(-1)
        ).view(C, self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1])

    def enable_cache(self):
        """Turn on WBMM-C: build M once from the *current* compact table and
        cache it, so every subsequent forward call skips the `index_select`
        gather. Call again (e.g. right after loading a different checkpoint)
        to refresh a stale cache. Works both before AND after a
        `cache=False` fusion (there is still a compact table to build from
        either way); no-op only once `cache=True` fusion has replaced it with
        a dense `M_fused` outright -- nothing left to cache from at that
        point, the block is already at its fastest, un-cacheable-further
        form."""
        if hasattr(self, 'M_fused'):
            return
        table = self._current_table()
        with torch.no_grad():
            self.M_cache = self._build_matrix(table.shape[0]).detach().clone()
        self.use_cache = True

    def disable_cache(self):
        """Revert to WBMM-NC: rebuild M from the compact table on every
        forward call."""
        self.use_cache = False
        self.M_cache = None

    def train(self, mode=True):
        # Re-entering training mode means gradients may flow through R
        # again, so a cached M would silently go stale -- drop it.
        if mode:
            self.disable_cache()
        return super().train(mode)

    def _current_matrix(self, C):
        """M for this forward call, cheapest source first: a fully dense
        `M_fused` (`cache=True` deploy) if one exists, else the cached
        (C,d,d) buffer if WBMM-C is active, else a fresh `index_select`
        against whichever compact table is currently authoritative
        (WBMM-NC -- either the raw `relative_position_bias_table` pre-deploy,
        or `R_fused` post-`cache=False` deploy)."""
        if hasattr(self, 'M_fused'):
            return self.M_fused
        if self.use_cache and self.M_cache is not None:
            return self.M_cache
        return self._build_matrix(C)

    def forward(self, x):
        if self.deployed:
            return self._forward_deployed(x)
        B, C, H, W = x.shape
        if H == self.window_size[0] and W == self.window_size[1]:
            # the whole feature map IS a single window (e.g. S4 at 7x7)
            if self.small_kernel == "True" and self.num_i == 3:
                # Y = WBMM(X) + BN(DW5(X)) + BN(DW3(X))  (paper Sec. 3.7)
                x2 = self.bn2(self.dwconv5(x)) + self.bn3(self.dwconv3(x))
                x = x.reshape(B, C, H * W).transpose(0, 1)
                x = x @ self._current_matrix(C) + x
                x = x.reshape(C, B, H, W).transpose(0, 1)
                x = self.bn1(x) + x2
            else:
                x = x.reshape(B, C, H * W).transpose(0, 1)
                x = x @ self._current_matrix(C) + x
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
            x = x @ self._current_matrix(C) + x
            x = x.reshape(C, B, H // self.window_size[0], W // self.window_size[1],
                          self.window_size[0], self.window_size[1]) \
                 .permute(1, 0, 2, 4, 3, 5).reshape(B, C, H, W)
        return x

    def _forward_deployed(self, x):
        """Inference-time path once `reparameterize_wbmm()` has been called:
        a single per-channel matmul + bias add, nothing else. `_current_matrix`
        transparently picks the dense `M_fused` (`cache=True` deploy) or
        indexes the compact `R_fused` (`cache=False` deploy, optionally
        itself cached via `enable_cache()`) -- the reshape/window logic below
        is identical either way."""
        B, C, H, W = x.shape
        wh, ww = self.window_size
        M = self._current_matrix(C)
        if H == wh and W == ww:
            x = x.reshape(B, C, H * W).transpose(0, 1)
            x = x @ M + self.bias_fused.view(C, 1, 1)
            x = x.reshape(C, B, H, W).transpose(0, 1)
        else:
            x = x.reshape(B, C, H // wh, wh, W // ww, ww) \
                 .permute(1, 0, 2, 4, 3, 5) \
                 .reshape(C, B * (H // wh) * (W // ww), wh * ww)
            x = x @ M + self.bias_fused.view(C, 1, 1)
            x = x.reshape(C, B, H // wh, W // ww, wh, ww) \
                 .permute(1, 0, 2, 4, 3, 5).reshape(B, C, H, W)
        return x

    @torch.no_grad()
    def _expand_to_dense(self):
        """Collapse the current compact `R_fused` into a dense `M_fused`
        (the `cache=True` / WBMM-C deploy target) and discard what inference
        no longer needs. Internal -- called by `reparameterize_wbmm(cache=True)`
        (either immediately, or later via a second `reparameterize_wbmm(cache=True)`
        call on a block that was first deployed with `cache=False`)."""
        C = self.bias_fused.shape[0]
        wh, ww = self.window_size
        d = wh * ww
        M_fused = torch.index_select(
            self.R_fused, 1, self.relative_position_index.view(-1)).view(C, d, d)
        self.register_buffer('M_fused', M_fused)
        del self.R_fused
        del self.relative_position_index
        self.use_cache = False
        self.M_cache = None

    @torch.no_grad()
    def reparameterize_wbmm(self, cache=False):
        """Fuse this operator's "+x" shortcut -- and, at S4 of WBMM-P/N, the
        Sec. 3.7 multi-kernel branch -- for inference. Two deploy targets,
        mirroring the paper's own WBMM-NC / WBMM-C split (Sec. 3.4.5) but
        applied to the *fused*, BatchNorm-free operator rather than the
        un-fused one:

        * `cache=False` (default -- "fused WBMM-NC"): the compact table
          R in R^{C x (2wh-1)(2ww-1)} is updated *in place* (same shape,
          renamed `R_fused`) and a new (C,) `bias_fused` is introduced.
          Every forward call still does `M = R_fused[:, I]` -- the exact
          same `index_select` training already does -- just against a
          table that already has the shortcut/BN/multi-kernel branch baked
          in, with the now-redundant BatchNorm and dwconv3/dwconv5
          sub-modules deleted outright. Folding a per-channel scale, an
          identity shortcut, or another depthwise kernel into R never
          changes R's *shape* (see `_depthwise_kernel_to_bias_table`), so
          this target has STRICTLY FEWER stored elements than the un-fused
          block: nothing bigger ever replaces what was deleted.
        * `cache=True` ("fused WBMM-C"): `R_fused` is additionally expanded,
          once, into a dense M_fused in R^{C x d x d} (d = wh*ww), and
          R_fused / the index are then discarded, removing `index_select`
          from every forward call entirely. This trades the compact
          `(2w-1)^2`-per-channel table for a dense `d^2`-per-channel matrix
          (7x7 window: 169 -> 2401 per channel, ~14x), which typically makes
          this target LARGER on disk than the un-fused block despite
          deleting the same BatchNorm parameters.

        Both targets are algebraically exact and equal each other: `M_fused`
        is nothing more than `R_fused` expanded through the very same index
        every un-fused forward call already uses, so both reproduce the
        training-time output identically up to floating-point associativity
        (see `tests/test_equivalence.py`, which checks both, against each
        other and against the pre-fusion reference).

        Idempotent, and safe on a model mixing already-fused and
        not-yet-fused blocks. Calling `reparameterize_wbmm(cache=True)` on a
        block already deployed with `cache=False` *upgrades* it in place
        (expands the surviving `R_fused`); the reverse (dense -> compact) is
        not supported, since `cache=True` discards R_fused, and there is
        nothing this method needs it for afterwards.
        """
        if self.deployed:
            if cache and hasattr(self, 'R_fused'):
                self._expand_to_dense()
            return
        C = self.relative_position_bias_table.shape[0]
        wh, ww = self.window_size
        center = (wh - 1) * (2 * ww - 1) + (ww - 1)   # flat index of relative offset (0, 0), c.f. Eq. 6-7

        R_fused = self.relative_position_bias_table.detach().clone()
        bias = torch.zeros(C, dtype=R_fused.dtype, device=R_fused.device)

        if self.small_kernel == "True" and self.num_i == 3:
            # BN1(WBMM(X) + X) + BN2(DW5(X)) + BN3(DW3(X)) -- fold all three
            # branches into ONE compact table + ONE bias (Sec. 3.7). Adding an
            # identity shortcut or another depthwise kernel to R only ever
            # touches offsets already inside R's own support, so this never
            # changes R's shape -- see `_depthwise_kernel_to_bias_table`.
            s1, t1 = _bn_scale_shift(self.bn1)
            R_fused[:, center] += 1.0                                     # "+x" shortcut, pre-BN1
            R_fused = R_fused * s1.view(C, 1)
            bias = bias + t1

            s2, t2 = _bn_scale_shift(self.bn2)
            R5 = _depthwise_kernel_to_bias_table(self.dwconv5.weight.data, self.window_size)
            R_fused = R_fused + R5 * s2.view(C, 1)
            bias = bias + t2

            s3, t3 = _bn_scale_shift(self.bn3)
            R3 = _depthwise_kernel_to_bias_table(self.dwconv3.weight.data, self.window_size)
            R_fused = R_fused + R3 * s3.view(C, 1)
            bias = bias + t3

            del self.dwconv3, self.dwconv5, self.bn1, self.bn2, self.bn3
        else:
            R_fused[:, center] += 1.0                                     # "+x" shortcut only

        del self.relative_position_bias_table
        self.register_buffer('R_fused', R_fused)
        self.register_buffer('bias_fused', bias)
        self.use_cache = False   # any pre-existing WBMM-C cache was built from R, now gone -- drop it
        self.M_cache = None
        self.deployed = True
        if cache:
            self._expand_to_dense()


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

        self.deployed = False   # flips to True after reparameterize_wbmm()

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

    @torch.no_grad()
    def reparameterize_wbmm(self, cache=False):
        """Fuse everything foldable inside this block for inference:

          1. whatever is *inside* `self.dwconv` (e.g. the Sec. 3.7
             multi-kernel branch, if `self.dwconv` is a `wbmm` operator with
             `small_kernel=="True", num_i==3`) is fused first, so `dwconv`
             becomes a single matmul-or-conv + bias operator;
          2. the block-level `self.norm` BatchNorm -- which wraps `dwconv`'s
             output whether `dwconv` is a `wbmm` operator ('W'), a plain
             depthwise Conv2d ('D' or an ablation kernel size), or Identity
             (kernel_size == 0) -- is folded into that operator;
          3. the `pwconv2` FFN projection's trailing BatchNorm is folded into
             its preceding Linear.

        `cache` is forwarded to `self.dwconv.reparameterize_wbmm()` unchanged
        (see `wbmm.reparameterize_wbmm` for what it selects between; it has
        no effect when `self.dwconv` isn't a `wbmm` operator). Step 2 below
        folds the outer BN's per-channel scale into whichever compact/dense
        table `self.dwconv` ends up with either way -- a per-channel scalar
        multiply commutes with `index_select`, so this is exact regardless
        of which of the two `self.dwconv` chose.

        `GRNwithNHWC`, `SEBlock` and the two `LayerNorm`s used elsewhere in
        the network are runtime (input-dependent) normalizations, NOT fixed
        affine maps, so they are intentionally left untouched -- folding them
        would change the function computed, not just its speed.

        After this call the block contains no BatchNorm modules at all.
        Idempotent (including re-calling with a different `cache` to
        upgrade an already-`cache=False`-deployed block, same as
        `wbmm.reparameterize_wbmm` itself).
        """
        if self.deployed:
            if cache and hasattr(self.dwconv, 'reparameterize_wbmm'):
                self.dwconv.reparameterize_wbmm(cache=True)   # upgrade R_fused -> M_fused only; no-op if already dense
            return

        # 1) fuse the operator inside dwconv (no-op if it has no such method,
        #    e.g. plain Conv2d or Identity)
        if hasattr(self.dwconv, 'reparameterize_wbmm'):
            self.dwconv.reparameterize_wbmm(cache=cache)

        # 2) fold the outer BN into dwconv
        if isinstance(self.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if isinstance(self.dwconv, wbmm):
                s, t = _bn_scale_shift(self.norm)
                C = self.dwconv.bias_fused.shape[0]
                if hasattr(self.dwconv, 'M_fused'):
                    self.dwconv.M_fused.mul_(s.view(C, 1, 1))      # dense (cache=True) deploy
                else:
                    self.dwconv.R_fused.mul_(s.view(C, 1))         # compact (cache=False) deploy
                self.dwconv.bias_fused.mul_(s).add_(t)
            elif isinstance(self.dwconv, nn.Conv2d):
                fw, fb = _fuse_conv_bn(self.dwconv, self.norm)
                new_conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels,
                                     self.dwconv.kernel_size, stride=self.dwconv.stride,
                                     padding=self.dwconv.padding, groups=self.dwconv.groups,
                                     bias=True)
                # assign the Parameters directly (not copy_) so the fused
                # module exactly matches fw/fb's dtype & device, even if the
                # source model runs in fp16/bf16/fp64 or lives on GPU.
                new_conv.weight = nn.Parameter(fw)
                new_conv.bias = nn.Parameter(fb)
                self.dwconv = new_conv
            self.norm = nn.Identity()

        # 3) fold pwconv2's Linear(bias=False) + BatchNorm2d -> Linear(bias=True)
        if len(self.pwconv2) == 3:
            linear, permute, bn = self.pwconv2[0], self.pwconv2[1], self.pwconv2[2]
            fw, fb = _fuse_linear_bn(linear, bn)
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight = nn.Parameter(fw)
            new_linear.bias = nn.Parameter(fb)
            self.pwconv2 = nn.Sequential(new_linear, permute)

        self.deployed = True


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
#   `force=True`: this file and segmentation/mmseg_custom/models/backbones/wbmm.py
#   both register a class named 'WBMMBackbone' into mmseg's *same* global
#   registry; if both ever get imported in one process (e.g. this file is
#   imported first for its timm factories, then the dedicated segmentation
#   backbone is imported/reloaded for actual det/seg training), the second
#   registration must be allowed to win, or mmcv's Registry raises
#   `KeyError: 'WBMMBackbone is already registered in models'`. The dedicated
#   segmentation file is the one actually referenced by segmentation configs.
if has_mmseg:
    @seg_BACKBONES.register_module(force=True)
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
