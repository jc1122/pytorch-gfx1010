"""
workarounds/__init__.py

Import this module to automatically apply all gfx1010 workarounds:

    import workarounds

Currently patches:
- torch.nn.BatchNorm2d        -> BatchNorm2dGFX1010
  (MIOpen BN kernel uses DPP row_bcast:15/31, not valid on gfx1010 RDNA1)

- torch.nonzero / Tensor.nonzero -> CPU fallback
  (rocprim DeviceSelect/DeviceReduce kernels not compiled for gfx1010
  because target_arch enum in rocprim doesn't include gfx1010)

- torch.masked_select / Tensor[bool_mask] -> CPU fallback
  (same root cause as nonzero — rocprim kernel missing for gfx1010)

- torch.Tensor.__repr__       -> CPU fallback for printing
  (tensor printing calls masked_select internally)
"""
import torch
import torch.nn as nn
from .batchnorm_gfx1010 import BatchNorm2dGFX1010

# ── BatchNorm2d ────────────────────────────────────────────────────────────────
nn.BatchNorm2d = BatchNorm2dGFX1010

# ── nonzero ───────────────────────────────────────────────────────────────────
# rocprim's device-level scan kernels (used by nonzero / masked_select /
# boolean indexing) are absent for gfx1010 because rocprim's target_arch enum
# only lists gfx1030, gfx1100, etc.  CPU fallback adds a round-trip but keeps
# the output tensor on the original device.

_orig_tensor_nonzero = torch.Tensor.nonzero

def _nonzero_gfx1010(self, *, as_tuple=False):
    if self.is_cuda:
        result = _orig_tensor_nonzero(self.cpu(), as_tuple=as_tuple)
        if as_tuple:
            return tuple(t.to(self.device) for t in result)
        return result.to(self.device)
    return _orig_tensor_nonzero(self, as_tuple=as_tuple)

torch.Tensor.nonzero = _nonzero_gfx1010

_orig_nonzero = torch.nonzero

def _torch_nonzero_gfx1010(input, *, out=None, as_tuple=False):
    if input.is_cuda:
        result = _orig_nonzero(input.cpu(), as_tuple=as_tuple)
        if as_tuple:
            return tuple(t.to(input.device) for t in result)
        return result.to(input.device)
    return _orig_nonzero(input, out=out, as_tuple=as_tuple)

torch.nonzero = _torch_nonzero_gfx1010

# ── masked_select ─────────────────────────────────────────────────────────────

_orig_masked_select = torch.masked_select

def _masked_select_gfx1010(input, mask):
    if input.is_cuda:
        return _orig_masked_select(input.cpu(), mask.cpu()).to(input.device)
    return _orig_masked_select(input, mask)

torch.masked_select = _masked_select_gfx1010

# ── boolean indexing: tensor[bool_mask] ───────────────────────────────────────

_orig_getitem = torch.Tensor.__getitem__

def _getitem_gfx1010(self, idx):
    if (self.is_cuda
            and isinstance(idx, torch.Tensor)
            and idx.dtype == torch.bool):
        return _orig_getitem(self.cpu(), idx.cpu()).to(self.device)
    return _orig_getitem(self, idx)

torch.Tensor.__getitem__ = _getitem_gfx1010

# ── tensor repr (print) ───────────────────────────────────────────────────────
# tensor printing calls masked_select internally; use CPU repr and re-add
# device info so the output still shows the correct device.

_orig_repr = torch.Tensor.__repr__

def _repr_gfx1010(self):
    if self.is_cuda:
        s = _orig_repr(self.cpu())
        # inject device= before closing paren, matching PyTorch repr style
        if s.endswith(")"):
            s = s[:-1] + f", device='{self.device}')"
        return s
    return _orig_repr(self)

torch.Tensor.__repr__ = _repr_gfx1010
