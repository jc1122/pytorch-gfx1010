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

- torch.scatter_add / Tensor.scatter_add[_] -> GPU index_add_ substitute
  for common dim-0 reductions, CPU fallback otherwise
  (native HIP scatter kernel can hit invalid-device-function on gfx1010)

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

# ── scatter_add ───────────────────────────────────────────────────────────────
# PyG aggregation uses Tensor.scatter_add_ for mean/sum reductions.  On gfx1010
# the native HIP scatter kernel can fail with hipErrorInvalidDeviceFunction.
# For the common PyG dim-0 reduction shape, route through index_add_, which
# runs natively on gfx1010. Fall back to CPU for less common scatter layouts.

_orig_scatter_add = torch.scatter_add
_orig_tensor_scatter_add = torch.Tensor.scatter_add
_orig_tensor_scatter_add_ = torch.Tensor.scatter_add_

def _scatter_add_args_to_cpu(input, index, src):
    return input.cpu(), index.cpu() if index.is_cuda else index, src.cpu() if src.is_cuda else src

def _scatter_add_index_for_dim0(index, src):
    if index.dim() == 1:
        return index
    if index.dim() != src.dim():
        return None
    if index.size(0) != src.size(0):
        return None
    if any(size == 0 for size in index.shape[1:]):
        return None
    if any(stride != 0 for stride in index.stride()[1:]):
        return None
    return index[(slice(None),) + (0,) * (index.dim() - 1)]

def _scatter_add_via_index_add(input, dim, index, src):
    dim = input.dim() + dim if dim < 0 else dim
    if dim != 0:
        return None
    if input.dim() != src.dim():
        return None
    if input.shape[1:] != src.shape[1:]:
        return None
    index_1d = _scatter_add_index_for_dim0(index, src)
    if index_1d is None:
        return None
    result = input.clone()
    result.index_add_(0, index_1d, src)
    return result

def _torch_scatter_add_gfx1010(input, dim, index, src, *, out=None):
    if input.is_cuda:
        result = _scatter_add_via_index_add(input, dim, index, src)
        if result is not None:
            if out is not None:
                out.copy_(result)
                return out
            return result
        cpu_input, cpu_index, cpu_src = _scatter_add_args_to_cpu(input, index, src)
        result = _orig_scatter_add(cpu_input, dim, cpu_index, cpu_src)
        result = result.to(input.device)
        if out is not None:
            out.copy_(result)
            return out
        return result
    return _orig_scatter_add(input, dim, index, src, out=out)

def _tensor_scatter_add_gfx1010(self, dim, index, src):
    if self.is_cuda:
        result = _scatter_add_via_index_add(self, dim, index, src)
        if result is not None:
            return result
        cpu_self, cpu_index, cpu_src = _scatter_add_args_to_cpu(self, index, src)
        return _orig_tensor_scatter_add(cpu_self, dim, cpu_index, cpu_src).to(self.device)
    return _orig_tensor_scatter_add(self, dim, index, src)

def _tensor_scatter_add_inplace_gfx1010(self, dim, index, src):
    if self.is_cuda:
        result = _scatter_add_via_index_add(self, dim, index, src)
        if result is not None:
            self.copy_(result)
            return self
        cpu_self, cpu_index, cpu_src = _scatter_add_args_to_cpu(self, index, src)
        result = _orig_tensor_scatter_add_(cpu_self, dim, cpu_index, cpu_src).to(self.device)
        self.copy_(result)
        return self
    return _orig_tensor_scatter_add_(self, dim, index, src)

torch.scatter_add = _torch_scatter_add_gfx1010
torch.Tensor.scatter_add = _tensor_scatter_add_gfx1010
torch.Tensor.scatter_add_ = _tensor_scatter_add_inplace_gfx1010

# ── LSTM / GRU (MIOpen CK JIT fails on gfx1010) ──────────────────────────────
# MIOpen's composable_kernel JIT path for RNN doesn't recognise gfx1010 and
# raises a compile error.  Disabling cudnn makes PyTorch fall back to its own
# pure-HIP RNN implementation which compiles correctly for gfx1010.
torch.backends.cudnn.enabled = False
