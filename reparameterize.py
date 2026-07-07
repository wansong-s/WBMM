#!/usr/bin/env python3
# --------------------------------------------------------
# WBMM: Windowed Batch Matrix Multiplication
# Code  : https://github.com/wansong-s/WBMM
# Licensed under the Apache 2.0 License [see LICENSE for details]
#
# reparameterize.py  (image classification)
# ==========================================
# Converts a trained WBMM classification checkpoint into an inference-only
# one where every foldable BatchNorm has been fused into the operator that
# feeds it:
#   * Sec. 3.7 multi-kernel branch (S4 of WBMM-P / WBMM-N):
#       BN1(WBMM(x)+x) + BN2(DW5(x)) + BN3(DW3(x))  ->  one table/matmul + bias
#   * every other WBMM ('W') block:
#       BN( WBMM(x) + x )                            ->  one table/matmul + bias
#   * 3x3 (or other odd-kernel) depthwise ('D') blocks:
#       BN( DWConv(x) )                               ->  one Conv2d(bias=True)
#   * the pwconv2 FFN projection:
#       BN( Linear(x) )                                ->  one Linear(bias=True)
#
# Every fusion is *lossless* -- the resulting model reproduces the original,
# un-fused model's output bit-for-bit up to floating point round-off. See
# tests/test_equivalence.py for the numerical proof on the real models.
#
# Two fused deploy targets (Sec. 3.4.5's WBMM-NC / WBMM-C split, applied to
# the *fused* model -- see `wbmm_reparam.py`'s `wbmm.reparameterize_wbmm`
# docstring for the full derivation):
#
#   default (no extra flag) -- deploy_nocache (WBMM-NC), "fused, compact":
#       BatchNorm / shortcut / multi-kernel fold straight into the EXISTING
#       compact (C,(2w-1)^2) relative-position table (-> `R_fused`, same
#       shape). A forward call still does one `index_select` -- exactly like
#       an un-fused model -- just against a smaller, BatchNorm-free one:
#       nothing bigger ever replaces what fusion deletes, so this format is
#       ALWAYS SMALLER than the input checkpoint.
#   `--cache` -- deploy_cache (WBMM-C), "fused, expanded":
#       `R_fused` is additionally expanded, once, into a dense (C,d,d)
#       `M_fused`, removing `index_select` from every forward call entirely.
#       This trades the compact `(2w-1)^2`-per-channel table for a dense
#       `d^2`-per-channel matrix (7x7 window: 169 -> 2401 per channel, ~14x),
#       which typically makes this format LARGER than the input checkpoint
#       despite deleting the same BatchNorm parameters.
#
# Usage
# -----
#   python reparameterize.py --model wbmm_t \
#       --checkpoint wbmm_t_in1k_224_w7.pth \
#       --output     wbmm_t_in1k_224_w7_deploy_nocache.pth \
#       --check
#
#   python reparameterize.py --model wbmm_t \
#       --checkpoint wbmm_t_in1k_224_w7.pth \
#       --output     wbmm_t_in1k_224_w7_deploy_cache.pth \
#       --cache --check
#
# `--half` addresses file size directly, independent of which format above
# you chose: it stores the output checkpoint's floating-point tensors as
# float16 (integer/bool buffers are left alone). Loading is unaffected:
# `load_any_checkpoint` upcasts back to float32 automatically either way, so
# inference still runs in fp32; add --check to see the (small but non-zero,
# larger than fusion's own residual) extra rounding this trades in for the
# smaller file before deciding whether it's worth it for your use case:
#   python reparameterize.py --model wbmm_t \
#       --checkpoint wbmm_t_in1k_224_w7.pth \
#       --output     wbmm_t_in1k_224_w7_deploy_fp16.pth \
#       --half --check
#
# Loading any of the checkpoints this script produces -- or an ordinary,
# un-converted one -- is one call, regardless of which of the three it is:
#
#   from reparameterize import load_any_checkpoint
#   model, fmt = load_any_checkpoint('wbmm_t', 'wbmm_t_in1k_224_w7_deploy_nocache.pth', num_classes=1000)
#
# `load_any_checkpoint` inspects the checkpoint's own keys
# (`probe_checkpoint_format`) to tell a plain / deploy_nocache / deploy_cache
# checkpoint apart and builds the matching architecture automatically. If
# you'd rather do it by hand for a *fused* checkpoint specifically, the fused
# architecture is `build_deploy_model()` (pass `cache=True` for the
# deploy_cache one):
#
#   model = build_deploy_model('wbmm_t', num_classes=1000)
#   model.load_state_dict(torch.load('wbmm_t_in1k_224_w7_deploy_nocache.pth')['model'])
#
# This script (and tests/test_equivalence.py) import the model from
# `wbmm_reparam.py`, not `wbmm.py` -- see that file's module docstring for
# why the two are interchangeable for every checkpoint `main.py` produces.
# --------------------------------------------------------
import argparse
import os

import torch
from timm.models import create_model

import wbmm_reparam  # noqa: F401  (registers wbmm_p / wbmm_n / wbmm_t / wbmm_s / *_dense with timm)


def reparameterize_model(model: torch.nn.Module, cache: bool = False) -> torch.nn.Module:
    """Fuse every foldable BatchNorm in `model` into the WBMM table/matrix /
    Conv2d / Linear that feeds it. In-place (also returned for convenience).

    `model` must be in eval() mode: fusion reads BatchNorm's *running*
    statistics, which are only meaningful outside of training mode.

    `cache` selects which of the two fused deploy targets every WBMM
    operator lands in (see the module docstring above and
    `wbmm_reparam.wbmm.reparameterize_wbmm` for the full derivation):
      * `cache=False` (default): fused, COMPACT -- BatchNorm/shortcut/
        multi-kernel fold into the existing relative-position table
        (`R_fused`, same shape as before), so `index_select` is still used
        at inference. Strictly fewer stored elements than the un-fused model.
      * `cache=True`: fused, DENSE -- `R_fused` is additionally expanded into
        a dense per-window matrix (`M_fused`), removing `index_select`
        entirely at the cost of usually being larger than the un-fused model.
    Both are algebraically exact and agree with each other bit-for-bit up to
    floating-point associativity (see tests/test_equivalence.py).

    Safe to call on any WBMM model (Pico/Nano/Tiny/Small, classification or
    dense-prediction variant): every module that knows how to fuse itself
    exposes a `reparameterize_wbmm(cache=...)` method, and this simply calls
    it on each of them (parents trigger their children first when relevant,
    and every method is idempotent, so the exact traversal order does not
    matter). Calling this again with `cache=True` on a model already fused
    with `cache=False` upgrades every block from compact to dense in place.
    """
    assert not model.training, "call model.eval() before reparameterize_model()"
    for m in model.modules():
        if hasattr(m, 'reparameterize_wbmm'):
            m.reparameterize_wbmm(cache=cache)
    return model


def build_deploy_model(model_name: str, cache: bool = False, **kwargs) -> torch.nn.Module:
    """Construct a freshly-initialised model already in its *post-fusion*
    (BatchNorm-free) architecture, ready to `load_state_dict()` a checkpoint
    produced by this script. The random weights used to reach that
    architecture are immediately overwritten by `load_state_dict`, so their
    actual values don't matter -- only the resulting module *structure* does.

    `cache` must match how the checkpoint being loaded was produced: `False`
    for the default fused-compact format (`R_fused` + `bias_fused` per WBMM
    block), `True` for `--cache`'s fused-dense format (`M_fused` +
    `bias_fused`) -- `load_any_checkpoint` picks this automatically from the
    checkpoint's own keys, so prefer that unless you already know which one
    you have.
    """
    model = create_model(model_name, **kwargs)
    model.eval()
    reparameterize_model(model, cache=cache)
    return model


def _load_state_dict_from_checkpoint(path):
    if str(path).startswith('https'):
        ckpt = torch.hub.load_state_dict_from_url(path, map_location='cpu', check_hash=True)
    else:
        ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict'):
            if key in ckpt:
                return ckpt[key]
    return ckpt   # already a raw state_dict


# ===========================================================================
#  Auto-detecting a checkpoint's format on load (plain / fused-compact /
#  fused-dense), from its keys alone -- no model construction needed first.
# ===========================================================================

def probe_checkpoint_format(state_dict: dict) -> str:
    """Inspect a WBMM classification checkpoint's *keys* -- no model
    construction needed -- and report which on-disk format it is, so the
    matching architecture can be built before `load_state_dict`. Returns:
      'deploy_cache'   -- fused, dense (`reparameterize_model(cache=True)` /
                          `--cache`): `M_fused` + `bias_fused`, no
                          `relative_position_bias_table`, no BatchNorm. Must
                          be loaded into `build_deploy_model(..., cache=True)`.
      'deploy_nocache' -- fused, compact (`reparameterize_model()` default /
                          no extra flag): `R_fused` + `bias_fused`, no
                          `relative_position_bias_table`, no BatchNorm. Must
                          be loaded into `build_deploy_model(..., cache=False)`.
      'plain'   -- ordinary trained (un-fused) checkpoint: ordinary
                  architecture, load normally.
    """
    keys = list(state_dict.keys())
    has_M_fused = any(k == 'M_fused' or k.endswith('.M_fused') for k in keys)
    has_R_fused = any(k == 'R_fused' or k.endswith('.R_fused') for k in keys)
    has_R = any(k == 'relative_position_bias_table' or k.endswith('.relative_position_bias_table')
                for k in keys)
    if has_M_fused and not has_R:
        return 'deploy_cache'
    if has_R_fused and not has_R:
        return 'deploy_nocache'
    return 'plain'


def load_any_checkpoint(model_name: str, checkpoint_path: str, **model_kwargs):
    """Build a WBMM classification model in whichever architecture matches
    `checkpoint_path`, load it, and return `(model, format_string)` -- so
    that the *same* call works for a plain checkpoint, a fused-compact
    checkpoint, and a fused-dense (`--cache`) checkpoint, without the caller
    needing to know in advance which one a given `.pth` file is
    (`probe_checkpoint_format`, above, is what makes this automatic).
    `model_kwargs` (e.g. `num_classes`, `window_size`, `drop_path_rate`) are
    forwarded to `create_model` / `build_deploy_model` exactly as they would
    be for an ordinary (un-fused) load.

    Raises `RuntimeError` on any real key/shape mismatch, exactly like a
    strict `load_state_dict()` would.
    """
    state_dict = _load_state_dict_from_checkpoint(checkpoint_path)
    fmt = probe_checkpoint_format(state_dict)

    if fmt in ('deploy_cache', 'deploy_nocache'):
        model = build_deploy_model(model_name, cache=(fmt == 'deploy_cache'), **model_kwargs)
        model.load_state_dict(state_dict)   # strict: architectures must match exactly
    else:
        model = create_model(model_name, **model_kwargs)
        model.eval()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"load_any_checkpoint({checkpoint_path!r}): detected format={fmt!r} but the state_dict "
                f"still doesn't match that architecture -- missing={missing}, unexpected={unexpected}. "
                f"(Common cause: --window_size doesn't match what the checkpoint was trained/converted "
                f"with.)")
    return model, fmt


def main():
    parser = argparse.ArgumentParser(
        description="Fuse a trained WBMM classification checkpoint's BatchNorm "
                    "layers into its WBMM table/matrix / conv / linear weights for faster inference.")
    parser.add_argument('--model', required=True,
                        help="model name registered with timm, e.g. wbmm_p / wbmm_n / wbmm_t / wbmm_s "
                             "/ wbmm_t_dense / wbmm_s_dense")
    parser.add_argument('--checkpoint', required=True, help="path to the trained (un-fused) checkpoint")
    parser.add_argument('--output', required=True, help="where to save the resulting checkpoint")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=7,
                        help="window size the checkpoint was trained with (square window; default 7, "
                             "the setting used by every released checkpoint)")
    parser.add_argument('--cache', action='store_true',
                        help="fuse into the DENSE (C,d,d) WBMM-C matrix (M_fused) instead of the default "
                             "COMPACT (C,(2w-1)^2) WBMM-NC table (R_fused). The default usually SHRINKS "
                             "the checkpoint versus --checkpoint (BatchNorm disappears, nothing bigger "
                             "replaces it); --cache usually GROWS it (compact-to-dense blow-up) but "
                             "removes index_select from every forward call. Both are exact and agree "
                             "with each other -- see --check.")
    parser.add_argument('--check', action='store_true',
                        help="run a forward pass on random input before/after fusion and report the "
                             "max output difference, as a sanity check")
    parser.add_argument('--half', action='store_true',
                        help="store the output checkpoint's floating-point tensors as float16 instead of "
                             "float32 (integer/bool buffers are left alone). Roughly halves the --output "
                             "file. Purely a storage change: load_state_dict() upcasts float16 tensors "
                             "back to float32 on load regardless, so load_any_checkpoint / "
                             "build_deploy_model need no changes and inference still runs in fp32. The "
                             "real cost is precision -- every weight is rounded to float16 once, on top "
                             "of (not instead of) fusion's own much smaller residual -- pass --check to "
                             "see exactly how much that adds before deciding it's worth the size.")
    args = parser.parse_args()

    model = create_model(args.model, num_classes=args.num_classes,
                        window_size=(args.window_size, args.window_size))

    state_dict = _load_state_dict_from_checkpoint(args.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading checkpoint: {unexpected}")
    model.eval()

    n_params_before = sum(p.numel() for p in model.parameters())

    if args.check:
        x = torch.randn(2, 3, args.window_size * 32, args.window_size * 32)
        with torch.no_grad():
            y_before = model(x).clone()

    reparameterize_model(model, cache=args.cache)
    fmt_label = 'deploy_cache' if args.cache else 'deploy_nocache'

    n_params_after = sum(p.numel() for p in model.parameters())

    if args.check:
        with torch.no_grad():
            y_after = model(x)
        err = (y_before - y_after).abs().max().item()
        action = {"deploy_cache": "dense fusion", "deploy_nocache": "compact fusion"}[fmt_label]
        print(f"[check] max |output difference| before vs after {action}: {err:.3e}  "
              f"(should be ~1e-6 or smaller, i.e. float rounding only)")

    out_state_dict = model.state_dict()

    if args.half:
        out_state_dict = {k: (v.half() if torch.is_floating_point(v) else v)
                          for k, v in out_state_dict.items()}

    torch.save({'model': out_state_dict}, args.output)
    file_size_before = os.path.getsize(args.checkpoint) if os.path.exists(args.checkpoint) else None
    file_size_after = os.path.getsize(args.output)
    n_stored_before = sum(t.numel() for t in state_dict.values())
    n_stored_after = sum(t.numel() for t in out_state_dict.values())

    print(f"Saved {fmt_label!r} checkpoint to {args.output}")
    print(f"  keys in state_dict:      {len(state_dict)} -> {len(out_state_dict)}  "
          f"({'nothing from the input is kept AND duplicated -- ' if len(out_state_dict) <= len(state_dict) else ''}"
          f"see 'Why it changed' below for why fewer/equal keys can still mean a bigger file)")
    print(f"  nn.Parameter count:      {n_params_before/1e6:.3f}M -> {n_params_after/1e6:.3f}M")
    print(f"  total stored tensors:    {n_stored_before/1e6:.3f}M -> {n_stored_after/1e6:.3f}M elements "
          f"(parameters + buffers -- this is what actually ends up on disk; nn.Parameter count alone "
          f"can be a misleading proxy for it, see below)")
    if file_size_before is not None:
        print(f"  file size:               {file_size_before/1024**2:.1f} MB -> {file_size_after/1024**2:.1f} MB"
              f"{'  (float16 storage, --half)' if args.half else ''}")
    else:
        print(f"  file size:               {file_size_after/1024**2:.1f} MB"
              f"{'  (float16 storage, --half)' if args.half else ''}")

    if args.cache:
        print(f"  Why it changed: BatchNorm's affine parameters (small: 2 vectors of length C per block) "
              f"are gone -- folded into the WBMM matrix / conv / linear weights that feed them -- which "
              f"is why nn.Parameter count drops. Every `relative_position_bias_table` and BatchNorm key "
              f"is deleted outright, not kept alongside the new one (the key count above went down, not "
              f"up -- there is no double-storage here). But the WBMM block's *surviving* tensor, "
              f"M_fused, is a dense (C,d,d) matrix replacing R's compact (C,(2w-1)^2) table, and that "
              f"per-tensor expansion typically dominates the deletions: for a 7x7 window, "
              f"(2*7-1)^2=169 per channel becomes 49*49=2401 per channel, ~14x larger. Net effect: fewer "
              f"nn.Parameters, fewer keys, but usually more total stored elements and a bigger file -- "
              f"the documented memory-for-speed trade-off of turning windowed indexing into a single "
              f"dense matmul (see the README's 'Reparameterization for inference' section). If you "
              f"specifically want the file smaller than the original regardless, drop --cache (the "
              f"default fused-compact format is built precisely to stay smaller), or add --half on top "
              f"of either (see --help).")
    else:
        print(f"  Why it changed: BatchNorm's affine parameters (small: 2 vectors of length C per block), "
              f"and (at S4 of Pico/Nano) the extra dwconv3/dwconv5 kernels, are gone -- folded directly "
              f"into the SAME (C,(2w-1)^2) table R already had (now called R_fused, same shape), plus "
              f"one new small (C,) bias per WBMM block. Nothing bigger ever replaces what fusion deletes "
              f"here, so both nn.Parameter count and total stored elements should DECREASE versus "
              f"--checkpoint -- pass --cache instead if you want the alternative, usually-larger, "
              f"index_select-free dense format (see --help).")

    if args.check:
        reload_kwargs = dict(num_classes=args.num_classes, window_size=(args.window_size, args.window_size))
        reloaded_model, detected_fmt = load_any_checkpoint(args.model, args.output, **reload_kwargs)
        with torch.no_grad():
            y_reloaded = reloaded_model(x)
        err_reloaded = (y_before - y_reloaded).abs().max().item()
        expectation = ("expected to land around 1e-3 (can be a bit above or below) -- now dominated by "
                       "--half's float16 rounding, on top of fusion's own much smaller residual" if args.half else
                       "should be ~1e-6 or smaller -- float rounding only")
        print(f"[check] max |output difference| before-fusion vs a FRESH RELOAD of {args.output} "
              f"(auto-detected as {detected_fmt!r}): {err_reloaded:.3e}  ({expectation}; this proves "
              f"the *saved file* round-trips correctly through `load_any_checkpoint`, not just the "
              f"in-memory model from before it was written)")


if __name__ == '__main__':
    main()
