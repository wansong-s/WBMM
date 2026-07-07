# --------------------------------------------------------
# WBMM: Windowed Batch Matrix Multiplication
# Code  : https://github.com/wansong-s/WBMM
# Licensed under the Apache 2.0 License [see LICENSE for details]
#
# tests/test_equivalence.py  (image classification)
# ==================================================
# Numerical proof, as a test suite, that:
#
#   1. depthwise conv2d <=> position-dependent matmul (Theorem 3.2), and
#      kernel taps beyond (2H-1)x(2W-1) are provably inert (Theorem 3.1);
#   2. WBMM(window=w) is *exactly* a local (2w-1)x(2w-1) depthwise conv,
#      applied independently inside each non-overlapping window;
#   3. every `reparameterize_wbmm()` fusion in this repo -- the Sec. 3.7
#      multi-kernel branch, the per-block outer BatchNorm, and the pwconv2
#      FFN BatchNorm -- reproduces the corresponding *training-time* forward
#      pass bit-for-bit up to float rounding, on the REAL model code (not a
#      re-implementation), for BOTH fused deploy targets (`cache=False`'s
#      compact `R_fused` and `cache=True`'s dense `M_fused` -- see
#      `wbmm_reparam.wbmm.reparameterize_wbmm`'s docstring for the derivation);
#   4. the full classification models (WBMM-P/N/T/S) are end-to-end
#      equivalent before and after `reparameterize_model()` at both `cache`
#      settings;
#   5. a fused checkpoint saved to disk round-trips correctly through a
#      fresh `build_deploy_model(cache=...)` + `load_state_dict()`, for both
#      deploy targets, on a checkpoint that went through real
#      forward/backward/optimizer steps (not just freshly-initialised
#      weights); and that the shared relative-position-index cache actually
#      avoids recomputation;
#   6. `probe_checkpoint_format` / `load_any_checkpoint` correctly tell a
#      plain, a fused-compact ("deploy_nocache", `R_fused`), and a
#      fused-dense ("deploy_cache", `M_fused`) checkpoint apart from their
#      keys alone and reconstruct the matching architecture, and raise on a
#      genuinely corrupted one; and
#   7. the default (`cache=False`) fusion target always makes the checkpoint
#      SMALLER (BatchNorm disappears, nothing bigger replaces it), while
#      `cache=True` usually makes it LARGER (the compact table's (2w-1)^2
#      entries per channel expand into a dense d^2 matrix) -- the trade-off
#      `reparameterize.py --cache` documents is real, not just asserted.
#
# Scope note: this suite covers the classification model only (`wbmm_reparam.py`
# / root `reparameterize.py`). The detection and segmentation backbones under
# detection/ and segmentation/ are untouched by this change and are not
# exercised here. WBMM-C's separate inference-time weight-caching mechanism
# (`enable_cache()` / `disable_cache()`, a `--cache_only` checkpoint) is also
# out of scope for now -- it's a lighter-weight, fully-reversible alternative
# to the fusion covered here, not required to produce or load either deploy
# target above.
#
# Run with:  pytest tests/test_equivalence.py -v
#        or: python tests/test_equivalence.py         (no pytest required)
# --------------------------------------------------------
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wbmm_reparam as wbmm_module                                        # noqa: E402  (root, classification)
from reparameterize import (reparameterize_model, build_deploy_model,     # noqa: E402
                             probe_checkpoint_format, load_any_checkpoint)

torch.manual_seed(0)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _build_full_matrix(kernel, H, W):
    """Theorem 3.2's construction (Sec. 3.3), NOT windowed (proof Steps 1-3):
    embed a (C,1,Kh,Kw) depthwise kernel into a dense (C,H*W,H*W) matrix such
    that flattening a (B,C,H,W) input in raster order (t=h*W+w) and computing
    `x_flat @ M` reproduces zero-padded depthwise conv2d exactly (Eq. 2).
    Self-contained (no dependency beyond torch) so this proof lives directly
    in the test suite rather than in a separate reference package.

        M[c, t1, t2] = K[c, 0, m-h+kh, n-w+kw]   if the offset is in range
                     = 0                          otherwise
        where t1 = m*W+n indexes the INPUT position, t2 = h*W+w the OUTPUT
        position, and kh=Kh//2, kw=Kw//2.

    O(H^2 W^2) memory -- fine for this test's small H, W; see the paper's
    "Memory footprint" limitation (Sec. O) for why WBMM instead windows this
    construction rather than materialising it globally.
    """
    C, _, Kh, Kw = kernel.shape
    assert Kh % 2 == 1 and Kw % 2 == 1, "kernel height/width must be odd"
    kh, kw = Kh // 2, Kw // 2
    device = kernel.device

    hh, ww = torch.meshgrid(torch.arange(H, device=device),
                             torch.arange(W, device=device), indexing='ij')
    pos_h = hh.reshape(-1)          # (HW,) raster order, shared by input & output indices
    pos_w = ww.reshape(-1)

    delta_h = pos_h[:, None] - pos_h[None, :]     # m - h  -> (HW, HW)
    delta_w = pos_w[:, None] - pos_w[None, :]     # n - w
    mask = (delta_h.abs() <= kh) & (delta_w.abs() <= kw)

    row = (delta_h + kh).clamp(0, Kh - 1)
    col = (delta_w + kw).clamp(0, Kw - 1)
    flat_idx = (row * Kw + col).reshape(-1)

    kernel_flat = kernel.reshape(C, Kh * Kw)
    M = kernel_flat[:, flat_idx].view(C, H * W, H * W)
    M = M * mask.view(1, H * W, H * W)
    return M


def _randomize_bn(bn):
    with torch.no_grad():
        bn.running_mean.copy_(torch.randn_like(bn.running_mean) * 0.5)
        bn.running_var.copy_(torch.rand_like(bn.running_var) * 2.0 + 0.2)
        bn.weight.copy_(torch.randn_like(bn.weight) * 0.8 + 1.0)
        bn.bias.copy_(torch.randn_like(bn.bias) * 0.3)


def _pseudo_train(model, steps=4, batch=2, res=224, lr=0.05):
    """A handful of real forward/backward/optimizer steps so BatchNorm
    running stats and weights move off their (trivial) initial values --
    a meaningful stand-in for 'a trained checkpoint' without needing
    ImageNet."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(steps):
        x = torch.randn(batch, 3, res, res)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


# =======================================================================
# 1) Theorem 3.2: depthwise conv2d  ==  position-dependent matmul
# =======================================================================
def test_theorem_3_2_conv_equals_matmul():
    dtype = torch.float64
    for (H, W) in [(6, 6), (7, 7), (8, 5), (14, 14)]:
        for K in [1, 3, 5, 7]:
            if K > 2 * min(H, W) - 1:
                continue
            x = torch.randn(3, 4, H, W, dtype=dtype)
            kernel = torch.randn(4, 1, K, K, dtype=dtype) * 0.3
            bias = torch.randn(4, dtype=dtype)

            kh, kw = K // 2, K // 2
            y_ref = F.conv2d(x, kernel, bias=bias, padding=(kh, kw), groups=4)

            M = _build_full_matrix(kernel, H, W)
            xf = x.reshape(3, 4, H * W).permute(1, 0, 2)
            y_mat = (xf @ M + bias.view(4, 1, 1)).permute(1, 0, 2).reshape(3, 4, H, W)

            err = (y_ref - y_mat).abs().max().item()
            assert err < 1e-9, f"Theorem 3.2 violated at H={H} W={W} K={K}: err={err}"


# =======================================================================
# 2) Theorem 3.1: kernel taps beyond (2H-1)x(2W-1) are provably inert
# =======================================================================
def test_theorem_3_1_excess_kernel_is_inert():
    dtype = torch.float64
    H = W = 7
    K_eff = 2 * H - 1
    x = torch.randn(2, 4, H, W, dtype=dtype)
    kernel_eff = torch.randn(4, 1, K_eff, K_eff, dtype=dtype) * 0.3
    y_eff = F.conv2d(x, kernel_eff, padding=K_eff // 2, groups=4)

    K_bigger = K_eff + 6
    kernel_bigger = torch.randn(4, 1, K_bigger, K_bigger, dtype=dtype)  # random junk everywhere
    off = (K_bigger - K_eff) // 2
    kernel_bigger[:, :, off:off + K_eff, off:off + K_eff] = kernel_eff  # ...except the true kernel in the middle
    y_bigger = F.conv2d(x, kernel_bigger, padding=K_bigger // 2, groups=4)

    err = (y_eff - y_bigger).abs().max().item()
    assert err < 1e-9, "kernel taps beyond the max effective size should be provably inert"


# =======================================================================
# 3) Corollary: WBMM(window=w)  ==  local (2w-1)x(2w-1) conv per window
#    (uses the REAL `wbmm` operator class from wbmm_reparam.py, and -- to embed the
#    reference kernel into R -- the REAL `_depthwise_kernel_to_bias_table`
#    helper that `reparameterize_wbmm()`'s S4 branch also calls, rather than
#    a separate test-only re-implementation; the ground truth this test
#    checks against is `F.conv2d`, independent of either)
# =======================================================================
def _local_block_conv(x, kernel, window_size):
    B, C, H, W = x.shape
    wh, ww = window_size
    pad = kernel.shape[-1] // 2
    out = torch.zeros_like(x)
    for i in range(0, H, wh):
        for j in range(0, W, ww):
            blk = x[:, :, i:i + wh, j:j + ww]
            out[:, :, i:i + wh, j:j + ww] = F.conv2d(blk, kernel, padding=pad, groups=C) + blk
    return out


def test_window_equals_local_large_kernel():
    dtype = torch.float64
    C, B, n_h, n_w = 4, 2, 3, 2
    for w in [3, 5, 7, 14]:                     # 7 <-> 13 (UniRepLKNet's own optimum!), 14 <-> 27
        K = 2 * w - 1
        kernel = torch.randn(C, 1, K, K, dtype=dtype) * 0.3

        op = wbmm_module.wbmm(dim=C, small_kernel="False", num_i=0,
                              window_size=(w, w)).to(dtype).eval()
        with torch.no_grad():
            op.relative_position_bias_table.data.copy_(
                wbmm_module._depthwise_kernel_to_bias_table(kernel, (w, w)))

        x1 = torch.randn(B, C, w, w, dtype=dtype)
        err_single = (op(x1) - (F.conv2d(x1, kernel, padding=w - 1, groups=C) + x1)).abs().max().item()
        assert err_single < 1e-9

        x2 = torch.randn(B, C, n_h * w, n_w * w, dtype=dtype)
        y_wbmm = op(x2)
        err_multi = (y_wbmm - _local_block_conv(x2, kernel, (w, w))).abs().max().item()
        assert err_multi < 1e-9

        # contrast (must differ): WBMM is local/block-wise, not a true sliding conv
        y_global = F.conv2d(x2, kernel, padding=w - 1, groups=C) + x2
        err_global = (y_wbmm - y_global).abs().max().item()
        assert err_global > 1e-3, "windows should NOT behave like one giant sliding conv"


# =======================================================================
# 4) BatchNorm folding primitives (the same ones wbmm_reparam.py's
#    `reparameterize_wbmm()` uses -- tested directly off `wbmm_module` so
#    this exercises the actual in-model code, not a separate copy)
# =======================================================================
def test_bn_fusion_conv_and_linear():
    fuse_conv_bn = wbmm_module._fuse_conv_bn
    fuse_linear_bn = wbmm_module._fuse_linear_bn
    dtype = torch.float64
    C = 12

    for K, groups in [(3, C), (5, C), (1, 1)]:
        conv = nn.Conv2d(C, C, K, padding=K // 2, groups=groups, bias=False).to(dtype)
        bn = nn.BatchNorm2d(C).to(dtype)
        _randomize_bn(bn)
        conv.eval(); bn.eval()
        x = torch.randn(3, C, 9, 9, dtype=dtype)
        y_ref = bn(conv(x))
        fw, fb = fuse_conv_bn(conv, bn)
        y_fused = F.conv2d(x, fw, fb, padding=K // 2, groups=groups)
        assert (y_ref - y_fused).abs().max().item() < 1e-10

    linear = nn.Linear(40, C, bias=False).to(dtype)
    bn2 = nn.BatchNorm2d(C).to(dtype)
    _randomize_bn(bn2)
    linear.eval(); bn2.eval()
    x = torch.randn(2, 5, 5, 40, dtype=dtype)
    y_ref = bn2(linear(x).permute(0, 3, 1, 2))
    fw, fb = fuse_linear_bn(linear, bn2)
    y_fused = F.linear(x, fw, fb).permute(0, 3, 1, 2)
    assert (y_ref - y_fused).abs().max().item() < 1e-10


# =======================================================================
# 5) Sec. 3.7 multi-kernel fusion (S4 of WBMM-P / WBMM-N)
# =======================================================================
def test_multi_kernel_fusion():
    dtype = torch.float64
    C, w = 8, 7
    for cache in (False, True):
        op = wbmm_module.wbmm(dim=C, small_kernel="True", num_i=3, window_size=(w, w)).to(dtype)
        with torch.no_grad():
            op.relative_position_bias_table.copy_(torch.randn_like(op.relative_position_bias_table) * 0.4)
            op.dwconv3.weight.copy_(torch.randn_like(op.dwconv3.weight) * 0.5)
            op.dwconv5.weight.copy_(torch.randn_like(op.dwconv5.weight) * 0.5)
        for bn in [op.bn1, op.bn2, op.bn3]:
            _randomize_bn(bn)
        op.eval()

        x = torch.randn(5, C, w, w, dtype=dtype)
        with torch.no_grad():
            y_before = op(x).clone()
            op.reparameterize_wbmm(cache=cache)
            y_after = op(x)
        err = (y_before - y_after).abs().max().item()
        assert err < 1e-9, f"multi-kernel fusion mismatch (cache={cache}): {err}"
        assert hasattr(op, 'M_fused') == cache, f"cache={cache} should leave M_fused {'present' if cache else 'absent'}"
        # idempotent
        op.reparameterize_wbmm(cache=cache)
        with torch.no_grad():
            y_again = op(x)
        assert (y_after - y_again).abs().max().item() == 0.0


# =======================================================================
# 6) WBMMBlock-level fusion: outer BatchNorm + pwconv2, for 'W' and 'D'
# =======================================================================
def test_wbmm_block_fusion_W():
    dtype = torch.float64
    for cache in (False, True):
        for window in [(7, 7), (14, 14)]:
            block = wbmm_module.WBMMBlock(dim=16, small_kernel="False", num_i=1,
                                           kernel_size='W', window_size=window).to(dtype).eval()
            with torch.no_grad():
                block.dwconv.relative_position_bias_table.copy_(
                    torch.randn_like(block.dwconv.relative_position_bias_table) * 0.3)
            _randomize_bn(block.norm)
            _randomize_bn(block.pwconv2[2])

            x = torch.randn(2, 16, window[0] * 2, window[1] * 3, dtype=dtype)
            with torch.no_grad():
                y0 = block(x).clone()
                block.reparameterize_wbmm(cache=cache)
                y1 = block(x)
            err = (y0 - y1).abs().max().item()
            assert err < 1e-9, f"'W' block fusion mismatch at window={window}, cache={cache}: {err}"
            assert isinstance(block.norm, nn.Identity)
            assert len(block.pwconv2) == 2
            assert hasattr(block.dwconv, 'M_fused') == cache


def test_wbmm_block_fusion_D():
    dtype = torch.float64
    for cache in (False, True):
        block = wbmm_module.WBMMBlock(dim=16, small_kernel="False", num_i=1,
                                      kernel_size='D').to(dtype).eval()
        _randomize_bn(block.norm)
        _randomize_bn(block.pwconv2[2])
        x = torch.randn(2, 16, 15, 15, dtype=dtype)
        with torch.no_grad():
            y0 = block(x).clone()
            block.reparameterize_wbmm(cache=cache)
            y1 = block(x)
        err = (y0 - y1).abs().max().item()
        assert err < 1e-9, f"'D' block fusion mismatch (cache={cache}): {err}"


def test_wbmm_block_fusion_S4_multikernel():
    """The full stack: multi-kernel branch fusion (inside `wbmm`) followed by
    the outer-BN fold (inside `WBMMBlock`) -- 4 BatchNorms + 2 extra conv
    branches collapse into a single matmul + bias, at both `cache` settings."""
    dtype = torch.float64
    for cache in (False, True):
        block = wbmm_module.WBMMBlock(dim=24, small_kernel="True", num_i=3,
                                      kernel_size='W', window_size=(7, 7)).to(dtype).eval()
        with torch.no_grad():
            block.dwconv.relative_position_bias_table.copy_(
                torch.randn_like(block.dwconv.relative_position_bias_table) * 0.3)
            block.dwconv.dwconv3.weight.copy_(torch.randn_like(block.dwconv.dwconv3.weight) * 0.4)
            block.dwconv.dwconv5.weight.copy_(torch.randn_like(block.dwconv.dwconv5.weight) * 0.4)
        for bn in [block.dwconv.bn1, block.dwconv.bn2, block.dwconv.bn3, block.norm, block.pwconv2[2]]:
            _randomize_bn(bn)

        x = torch.randn(3, 24, 7, 7, dtype=dtype)
        with torch.no_grad():
            y0 = block(x).clone()
            block.reparameterize_wbmm(cache=cache)
            y1 = block(x)
        err = (y0 - y1).abs().max().item()
        assert err < 1e-9, f"S4 multi-kernel block fusion mismatch (cache={cache}): {err}"
        assert not hasattr(block.dwconv, 'dwconv3')   # sub-modules were dropped, not just bypassed
        assert hasattr(block.dwconv, 'M_fused') == cache


# =======================================================================
# 7) Full classification models: WBMM-P / N / T / S
# =======================================================================
def test_full_classification_models():
    for cache in (False, True):
        for name, ctor in [('wbmm_p', wbmm_module.wbmm_p), ('wbmm_n', wbmm_module.wbmm_n),
                           ('wbmm_t', wbmm_module.wbmm_t), ('wbmm_s', wbmm_module.wbmm_s)]:
            model = ctor()
            _pseudo_train(model, steps=3, batch=2)

            x_test = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                y_before = model(x_test).clone()

            reparameterize_model(model, cache=cache)

            with torch.no_grad():
                y_after = model(x_test)
            err = (y_before - y_after).abs().max().item()
            rel = err / y_before.abs().max().item()
            assert err < 1e-3, f"{name} (cache={cache}): end-to-end fusion mismatch: abs={err} rel={rel}"


# =======================================================================
# 8) Checkpoint round-trip: train -> save -> fuse -> save -> fresh reload
# =======================================================================
def test_checkpoint_roundtrip(tmp_path_or_str='/tmp'):
    for cache in (False, True):
        model = wbmm_module.wbmm_t(num_classes=10)
        _pseudo_train(model, steps=2, batch=2)

        trained_path = os.path.join(str(tmp_path_or_str), '_test_trained.pth')
        deploy_path = os.path.join(str(tmp_path_or_str), f'_test_deploy_{cache}.pth')
        torch.save({'model': model.state_dict()}, trained_path)

        x_test = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y_reference = model(x_test).clone()

        model2 = wbmm_module.wbmm_t(num_classes=10)
        sd = torch.load(trained_path, map_location='cpu')['model']
        missing, unexpected = model2.load_state_dict(sd, strict=True)
        assert not missing and not unexpected
        model2.eval()
        reparameterize_model(model2, cache=cache)
        torch.save({'model': model2.state_dict()}, deploy_path)

        model3 = build_deploy_model('wbmm_t', cache=cache, num_classes=10)
        deploy_sd = torch.load(deploy_path, map_location='cpu')['model']
        missing3, unexpected3 = model3.load_state_dict(deploy_sd, strict=True)
        assert not missing3 and not unexpected3

        with torch.no_grad():
            y_roundtrip = model3(x_test)
        err = (y_reference - y_roundtrip).abs().max().item()
        assert err < 1e-3, f"checkpoint round-trip mismatch (cache={cache}): {err}"


# =======================================================================
# 9) The relative-position-index cache is actually shared: a second block
#     built with the same window size must reuse the first block's I
#     instead of recomputing it (Sec. 3.4.2's I is a pure function of the
#     window geometry, see `_get_relative_position_index` in wbmm_reparam.py).
# =======================================================================
def test_shared_relative_position_index_cache():
    cache = wbmm_module._REL_POS_INDEX_CACHE
    probe_size = (11, 5)   # distinctive size, unused elsewhere in this file, so the test is order-independent
    assert probe_size not in cache, "test setup assumption violated: pick a window size unused elsewhere"

    op1 = wbmm_module.wbmm(dim=4, window_size=probe_size)
    assert probe_size in cache
    cached_tensor = cache[probe_size]

    op2 = wbmm_module.wbmm(dim=9, window_size=probe_size)   # different C, same window
    assert cache[probe_size] is cached_tensor, "a second block with the same window size must reuse the cached I"
    assert torch.equal(op1.relative_position_index, op2.relative_position_index)
    assert op1.relative_position_index.shape == (55, 55)   # 11*5 = 55


# =======================================================================
# 10) `probe_checkpoint_format`: pure key-inspection unit test, no model
#     construction or checkpoint I/O needed -- covers the three shapes a
#     real state_dict takes (see reparameterize.py):
#       'plain'           -- relative_position_bias_table present (ordinary / WBMM-NC)
#       'deploy_nocache'  -- R_fused present, R absent (fused, compact)
#       'deploy_cache'    -- M_fused present, R absent (fused, dense)
# =======================================================================
def test_probe_checkpoint_format():
    assert probe_checkpoint_format({
        'stages.0.0.dwconv.relative_position_bias_table': torch.zeros(4, 169),
        'stages.0.0.norm.weight': torch.zeros(4),
    }) == 'plain'

    assert probe_checkpoint_format({
        'stages.0.0.dwconv.R_fused': torch.zeros(4, 169),
        'stages.0.0.dwconv.bias_fused': torch.zeros(4),
    }) == 'deploy_nocache'

    assert probe_checkpoint_format({
        'stages.0.0.dwconv.M_fused': torch.zeros(4, 49, 49),
        'stages.0.0.dwconv.bias_fused': torch.zeros(4),
    }) == 'deploy_cache'

    # a model with zero WBMM ('W') blocks at all (e.g. an all-'D' ablation)
    # has none of these keys anywhere -- must default to 'plain', not crash
    assert probe_checkpoint_format({'downsample_layers.0.0.weight': torch.zeros(4, 3, 3, 3)}) == 'plain'


# =======================================================================
# 11) `load_any_checkpoint`: single entry point, three formats. Builds one
#     reference-trained model, derives all three checkpoint kinds from it
#     exactly as `reparameterize.py`'s CLI does, and checks that
#     `load_any_checkpoint` (a) reports the right format for each and
#     (b) reproduces the reference output for each; and that it still
#     raises on a genuinely corrupted state_dict rather than silently
#     tolerating arbitrary key mismatches.
# =======================================================================
def test_load_any_checkpoint_all_formats(tmp_path_or_str='/tmp'):
    reference = wbmm_module.wbmm_t(num_classes=10)
    _pseudo_train(reference, steps=2, batch=2)
    x_test = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y_reference = reference(x_test).clone()

    plain_path = os.path.join(str(tmp_path_or_str), '_test_load_any_plain.pth')
    torch.save({'model': reference.state_dict()}, plain_path)

    deploy_nocache_model = wbmm_module.wbmm_t(num_classes=10)
    deploy_nocache_model.load_state_dict(torch.load(plain_path, map_location='cpu')['model'])
    deploy_nocache_model.eval()
    reparameterize_model(deploy_nocache_model, cache=False)
    deploy_nocache_path = os.path.join(str(tmp_path_or_str), '_test_load_any_deploy_nocache.pth')
    torch.save({'model': deploy_nocache_model.state_dict()}, deploy_nocache_path)

    deploy_cache_model = wbmm_module.wbmm_t(num_classes=10)
    deploy_cache_model.load_state_dict(torch.load(plain_path, map_location='cpu')['model'])
    deploy_cache_model.eval()
    reparameterize_model(deploy_cache_model, cache=True)
    deploy_cache_path = os.path.join(str(tmp_path_or_str), '_test_load_any_deploy_cache.pth')
    torch.save({'model': deploy_cache_model.state_dict()}, deploy_cache_path)

    for path, expected_fmt in [(plain_path, 'plain'),
                               (deploy_nocache_path, 'deploy_nocache'), (deploy_cache_path, 'deploy_cache')]:
        model, fmt = load_any_checkpoint('wbmm_t', path, num_classes=10)
        assert fmt == expected_fmt, f"{path}: expected format {expected_fmt!r}, got {fmt!r}"
        with torch.no_grad():
            y = model(x_test)
        err = (y_reference - y).abs().max().item()
        assert err < 1e-3, f"load_any_checkpoint({path}) [{fmt}]: output mismatch {err}"

    # a real key/shape mismatch must still raise -- auto-detection is not a
    # license to silently tolerate corruption. Covers both code paths:
    # the 'plain' branch's manual missing/unexpected check, and the fused
    # branch's strict load_state_dict().
    for path, drop_suffix in [(plain_path, '.relative_position_bias_table'),
                              (deploy_nocache_path, '.bias_fused')]:
        bad_sd = dict(torch.load(path, map_location='cpu')['model'])
        bad_key = next(k for k in bad_sd if k.endswith(drop_suffix))
        del bad_sd[bad_key]
        bad_sd['totally_unexpected_key'] = torch.zeros(1)
        bad_path = os.path.join(str(tmp_path_or_str), '_test_load_any_bad.pth')
        torch.save({'model': bad_sd}, bad_path)
        try:
            load_any_checkpoint('wbmm_t', bad_path, num_classes=10)
            raise AssertionError(f"load_any_checkpoint should have raised on a corrupted {path}")
        except RuntimeError:
            pass


# =======================================================================
# 12) The two deploy targets are two STORAGE strategies for the same fused
#     function, not two different fusions: `cache=False`'s compact
#     `R_fused` and `cache=True`'s dense `M_fused` must each reproduce the
#     pre-fusion reference (already covered per-test above via the `cache`
#     loops), and therefore each other -- checked here directly, once, on
#     a full end-to-end model. And the storage direction the README /
#     `reparameterize.py --help` promise -- default shrinks, `--cache`
#     grows -- is asserted here numerically rather than only in prose.
# =======================================================================
def _count_stored_elements(model):
    """Total scalar count across the full state_dict (parameters + buffers)
    -- what actually determines a saved checkpoint's file size, unlike
    `model.parameters()` alone (which misses buffers such as R/M and
    BatchNorm running stats)."""
    return sum(v.numel() for v in model.state_dict().values())


def _find_wbmm_op(model):
    """First `wbmm` (the WBMM operator itself, not the surrounding block)
    instance in `model` -- used to inspect deploy-target attributes without
    hardcoding which stage/block happens to carry a 'W' kernel for a given
    config (e.g. wbmm_t's very first block is a 'D' block, not 'W')."""
    for m in model.modules():
        if isinstance(m, wbmm_module.wbmm):
            return m
    raise AssertionError("model has no 'W'-kernel WBMM operator to inspect")


def test_deploy_nocache_and_cache_agree():
    model = wbmm_module.wbmm_t(num_classes=10)
    _pseudo_train(model, steps=2, batch=2)
    model.eval()

    x_test = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y_reference = model(x_test).clone()

    model_nocache = copy.deepcopy(model)
    model_cache = copy.deepcopy(model)
    reparameterize_model(model_nocache, cache=False)
    reparameterize_model(model_cache, cache=True)
    assert not hasattr(_find_wbmm_op(model_nocache), 'M_fused')  # compact target: no dense matrix
    assert hasattr(_find_wbmm_op(model_cache), 'M_fused')        # dense target: has one

    with torch.no_grad():
        y_nocache = model_nocache(x_test)
        y_cache = model_cache(x_test)

    err_nocache = (y_reference - y_nocache).abs().max().item()
    err_cache = (y_reference - y_cache).abs().max().item()
    err_between = (y_nocache - y_cache).abs().max().item()
    assert err_nocache < 1e-3, f"no-cache fusion mismatch vs pre-fusion reference: {err_nocache}"
    assert err_cache < 1e-3, f"cache fusion mismatch vs pre-fusion reference: {err_cache}"
    assert err_between < 1e-3, f"no-cache and cache fusion disagree with each other: {err_between}"


def test_nocache_fusion_strictly_shrinks_storage():
    """The whole point of the default (`cache=False`) deploy target:
    BatchNorm (mean/var/weight/bias, 4 vectors of length C per BN layer)
    disappears and only one new small (C,) `bias_fused` buffer appears per
    WBMM block -- net negative. No compact-to-dense (2w-1)^2 -> d^2 blow-up
    happens on this path at all, so both the raw stored-element count and
    the trainable-parameter count must strictly shrink (or at worst stay
    equal for parameter count, since BN's running stats are buffers, not
    parameters, but BN's weight/bias ARE parameters and do shrink it)."""
    model = wbmm_module.wbmm_t(num_classes=10)
    _pseudo_train(model, steps=2, batch=2)
    model.eval()

    n_before = _count_stored_elements(model)
    n_params_before = sum(p.numel() for p in model.parameters())

    reparameterize_model(model, cache=False)

    n_after = _count_stored_elements(model)
    n_params_after = sum(p.numel() for p in model.parameters())

    assert n_after < n_before, (
        f"no-cache fusion should strictly shrink total stored elements: {n_before} -> {n_after}")
    assert n_params_after < n_params_before, (
        f"no-cache fusion should strictly shrink the trainable-parameter count: "
        f"{n_params_before} -> {n_params_after}")


def test_cache_fusion_grows_storage_for_window7():
    """The `cache=True` deploy target trades storage for speed: `R_fused`
    ((2w-1)^2 = 169 elements per channel at the paper's window=7) is
    additionally expanded into a dense `M_fused` (d^2 = 49^2 = 2401 elements
    per channel) -- a ~14x blow-up on that one tensor that dominates the
    (much smaller) BatchNorm removal in the other direction. Net effect:
    LARGER than even the original pre-fusion checkpoint, not merely larger
    than the no-cache target."""
    model = wbmm_module.wbmm_t(num_classes=10)
    _pseudo_train(model, steps=2, batch=2)
    model.eval()

    n_before = _count_stored_elements(model)
    reparameterize_model(model, cache=True)
    n_after = _count_stored_elements(model)

    assert n_after > n_before, (
        f"cache=True fusion should grow total stored elements for window=7: {n_before} -> {n_after}")


# =======================================================================
# stand-alone runner (no pytest required)
# =======================================================================
if __name__ == '__main__':
    tests = [
        test_theorem_3_2_conv_equals_matmul,
        test_theorem_3_1_excess_kernel_is_inert,
        test_window_equals_local_large_kernel,
        test_bn_fusion_conv_and_linear,
        test_multi_kernel_fusion,
        test_wbmm_block_fusion_W,
        test_wbmm_block_fusion_D,
        test_wbmm_block_fusion_S4_multikernel,
        test_full_classification_models,
        test_checkpoint_roundtrip,
        test_shared_relative_position_index_cache,
        test_probe_checkpoint_format,
        test_load_any_checkpoint_all_formats,
        test_deploy_nocache_and_cache_agree,
        test_nocache_fusion_strictly_shrinks_storage,
        test_cache_fusion_grows_storage_for_window7,
    ]
    n_ok = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"[PASS] {t.__name__}  ({time.time()-t0:.1f}s)")
            n_ok += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{n_ok}/{len(tests)} tests passed.")
    if n_ok != len(tests):
        sys.exit(1)
