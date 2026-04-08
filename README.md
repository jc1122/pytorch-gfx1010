# PyTorch gfx1010 (RX 5700 XT / RDNA1)

Patches and build script to compile **PyTorch 2.9.1** for AMD gfx1010 (Navi 10, RX 5700/5700 XT).

gfx1010 is not in any official PyTorch ROCm wheel. These patches fix compilation errors in the
[composable_kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) submodule.

For matmul support (rocBLAS), see the companion repo: [rocblas-gfx1010](https://github.com/jc1122/rocblas-gfx1010).

## Requirements

- ROCm 6.4 installed (`/opt/rocm`)
- Python 3.12 + venv
- ~125 GB free disk, 32 GB RAM recommended
- 32-core CPU (adjust `MAX_JOBS` in build script)
- Ubuntu 24.04 (Noble)

## What works after build + rocBLAS patch

| Operation | Status | Notes |
|-----------|--------|-------|
| Element-wise ops (add, mul, relu, etc.) | OK | |
| Reductions (sum, mean, max, norm) | OK | |
| Memory allocation / CPU-GPU transfer | OK | |
| matmul / nn.Linear (float32/64/16/bf16) | OK | Requires [rocblas-gfx1010](https://github.com/jc1122/rocblas-gfx1010) |
| Batched matmul (bmm) | OK | Requires rocblas-gfx1010 |
| Conv2d forward + backward (MIOpen) | OK | |
| LayerNorm, GroupNorm | OK | |
| scaled_dot_product_attention | OK | |
| AdamW / SGD optimizer step | OK | |
| BatchNorm2d forward / inference | OK | |
| BatchNorm2d backward (training) | WORKAROUND | MIOpen BN backward uses DPP row_bcast:15/31, not valid on gfx1010 |

## BatchNorm2d workaround

MIOpen's BN backward kernel uses `v_add_f32 ... row_bcast:15/31` (GCN-era DPP subgroup
broadcasts not available on gfx1010 RDNA1). Forward pass works fine; only the backward
pass (gradient computation during training) fails with `miopenStatusUnknownError`.

A drop-in replacement is provided in `workarounds/batchnorm_gfx1010.py`:

```python
from workarounds.batchnorm_gfx1010 import BatchNorm2dGFX1010 as BatchNorm2d
# Use exactly like nn.BatchNorm2d -- same constructor args, same behavior
```

Alternatively, `nn.GroupNorm` works natively and is often preferred for small batch sizes.

## Build instructions

```bash
# 1. Clone PyTorch v2.9.1
mkdir ~/pytorch-build && cd ~/pytorch-build
git clone --depth 1 --branch v2.9.1 https://github.com/pytorch/pytorch
git -C pytorch submodule update --init --recursive

# 2. Apply patches
patch -p1 -d pytorch/third_party/composable_kernel < patches/composable_kernel-gfx1010.patch

# 3. Create venv and build
python3 -m venv .venv
source .venv/bin/activate
pip install cmake ninja setuptools wheel typing_extensions pyyaml requests numpy

# Edit build.sh to set VENV and SRC paths, then:
bash build.sh
```

## Patches summary

### composable_kernel -- `ck.hpp` and `ck_tile/core/config.hpp`

1. **Buffer resource 3rd dword**: Added `__gfx101__` to the `gfx103` branch so gfx1010 gets
   the correct RDNA buffer addressing value (`0x31014000`).

2. **FMA instruction**: Added a separate `__gfx101__` case that only defines `CK_USE_AMD_V_FMAC_F32`.
   gfx1010 supports `v_fmac_f32` but does **not** support dot product instructions
   (`dot1-insts`, `dot10-insts`) -- those are RDNA2+ only.
