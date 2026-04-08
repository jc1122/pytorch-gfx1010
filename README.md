# PyTorch gfx1010 (RX 5700 XT / RDNA1)

Patches and build script to compile **PyTorch 2.9.1** for AMD gfx1010 (Navi 10, RX 5700/5700 XT).

gfx1010 is not in any official PyTorch ROCm wheel. These patches fix compilation errors in the
[composable_kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel) submodule.

## Requirements

- ROCm 6.4 installed (`/opt/rocm`)
- Python 3.12 + venv
- ~125 GB free disk, 32 GB RAM recommended
- 32-core CPU (adjust `MAX_JOBS` in build script)
- Ubuntu 24.04 (Noble)

## What works after build

| Operation | Status |
|-----------|--------|
| Element-wise ops (add, mul, relu, etc.) | ✅ |
| Reductions (sum, mean, etc.) | ✅ |
| Memory allocation / transfer | ✅ |
| matmul / nn.Linear via rocBLAS | ❌ (rocBLAS 6.4 has no gfx1010 Tensile kernels) |

To get matmul working, rocBLAS must also be built from source targeting gfx1010.

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

### composable_kernel — `ck.hpp` and `ck_tile/core/config.hpp`

1. **Buffer resource 3rd dword**: Added `__gfx101__` to the `gfx103` branch so gfx1010 gets
   the correct RDNA buffer addressing value (`0x31014000`).

2. **FMA instruction**: Added a separate `__gfx101__` case that only defines `CK_USE_AMD_V_FMAC_F32`.
   gfx1010 supports `v_fmac_f32` but does **not** support dot product instructions
   (`dot1-insts`, `dot10-insts`) — those are RDNA2+ only.
