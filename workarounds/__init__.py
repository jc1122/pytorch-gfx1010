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
  (same root cause as nonzero — rocprim kernel missing for gfx1010.
  Handles both direct boolean indexing and combined slicing/masking.)

- torch.Tensor.__repr__       -> CPU fallback for printing
  (tensor printing calls masked_select internally)

- torch.unique / Tensor.unique -> CPU fallback
  (same rocprim kernel issue; radix-sort / scan path hits missing gfx1010 binary)

- torch.backends.cudnn.enabled = False  (applied at import time)
  (MIOpen's composable_kernel JIT compilation for LSTM/GRU fails on gfx1010:
  CK config.hpp doesn't recognise gfx1010, raises "Need to define (only) one GPU
  target". Disabling cudnn makes LSTM/GRU fall back to PyTorch's own implementation.)
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
# Handles both direct: tensor[mask] and combined: tensor[:, mask]
# rocprim DeviceSelect kernel missing; CPU fallback.

_orig_getitem = torch.Tensor.__getitem__

def _has_cuda_bool_tensor(idx):
    if isinstance(idx, torch.Tensor):
        return idx.is_cuda and idx.dtype == torch.bool
    if isinstance(idx, tuple):
        return any(_has_cuda_bool_tensor(x) for x in idx)
    return False

def _to_cpu(idx):
    if isinstance(idx, torch.Tensor) and idx.is_cuda:
        return idx.cpu()
    if isinstance(idx, tuple):
        return tuple(_to_cpu(x) for x in idx)
    if isinstance(idx, list):
        return [_to_cpu(x) for x in idx]
    return idx

def _getitem_gfx1010(self, idx):
    if self.is_cuda and _has_cuda_bool_tensor(idx):
        return _orig_getitem(self.cpu(), _to_cpu(idx)).to(self.device)
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

# ── unique ────────────────────────────────────────────────────────────────────
# torch.unique uses a radix-sort/scan path via rocprim; same missing gfx1010
# kernel binary.  CPU fallback, result moved back to original device.

_orig_unique = torch.unique

def _unique_gfx1010(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if input.is_cuda:
        result = _orig_unique(input.cpu(), sorted=sorted,
                              return_inverse=return_inverse,
                              return_counts=return_counts,
                              dim=dim)
        if isinstance(result, tuple):
            return tuple(t.to(input.device) for t in result)
        return result.to(input.device)
    return _orig_unique(input, sorted=sorted, return_inverse=return_inverse,
                        return_counts=return_counts, dim=dim)

torch.unique = _unique_gfx1010

_orig_tensor_unique = torch.Tensor.unique

def _tensor_unique_gfx1010(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
    return _unique_gfx1010(self, sorted=sorted, return_inverse=return_inverse,
                           return_counts=return_counts, dim=dim)

torch.Tensor.unique = _tensor_unique_gfx1010

# ── LSTM / GRU (MIOpen CK JIT fails on gfx1010) ──────────────────────────────
# MIOpen's composable_kernel JIT path for RNN doesn't recognise gfx1010 and
# raises a compile error.  Disabling cudnn makes PyTorch fall back to its own
# pure-HIP RNN implementation which compiles correctly for gfx1010.
torch.backends.cudnn.enabled = False
