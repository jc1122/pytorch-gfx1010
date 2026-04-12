# PyTorch gfx1010 (RX 5700 XT / RDNA1)

Patches and build script to compile **PyTorch 2.9.1** for AMD gfx1010 (Navi 10, RX 5700/5700 XT).

Run the torch installer only. It installs the required gfx1010 rocBLAS layer automatically.

## Install

Use the installer:

```bash
curl -sSL https://raw.githubusercontent.com/jc1122/pytorch-gfx1010/main/install.sh | bash
```

The installer will auto-install the required gfx1010 rocBLAS runtime if it is missing.
On a first-time setup this may require `sudo`, because the rocBLAS runtime is installed
system-wide into `/opt/rocm`.

Do **not** install the release wheel by itself unless you also know how to install both the
gfx1010 rocBLAS runtime and the gfx1010 PyTorch workaround/autoload pieces manually. The
wheel alone does not enable the full runtime fixes for gfx1010. If you do install the wheel
directly, first CUDA use now fails fast with a clear error explaining that the gfx1010
rocBLAS runtime is missing and pointing you to the supported installer.

## Status

**PyTorch + rocBLAS are fully functional on gfx1010** for practical deep learning:
matmul, Conv2d, LayerNorm, GroupNorm, attention, AdamW — all work out of the box.

`install.sh` handles both layers of compatibility:

- it installs the gfx1010 rocBLAS runtime if needed
- it installs the PyTorch workaround package and autoload hook

After install, user code does not need to import anything extra.

This is not a hardware limitation — all required arithmetic works on gfx1010. The remaining
problems are mostly rocPRIM selection/sort gaps for gfx1010, and the installed workarounds
patch those paths automatically inside the Python environment.

---

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
| nonzero / masked_select / bool indexing / unique / tensor repr | AUTO-WORKAROUND | CPU fallback, result moved back to CUDA |
| scatter_add / PyG sum+mean aggregation | NATIVE PATCH | Common `dim=0` path uses compiled GPU `index_add_`; unsupported layouts still fall back through the Python workaround |
| LSTM / GRU | NATIVE PATCH | gfx1010 skips broken MIOpen RNN JIT and uses PyTorch's native GPU implementation |
| BatchNorm2d forward / inference | OK | |
| BatchNorm2d backward (training) | NATIVE PATCH | gfx1010 skips broken MIOpen BN kernels and uses PyTorch's native GPU implementation |

## BatchNorm2d and RNN native patches

MIOpen's BN training kernels use `v_add_f32 ... row_bcast:15/31` (GCN-era DPP subgroup
broadcasts not available on gfx1010 RDNA1), and MIOpen's RNN JIT path does not recognise
gfx1010 in its CK config. The native PyTorch patch skips those MIOpen paths on gfx1010:
BatchNorm uses PyTorch's native GPU BatchNorm kernels, and LSTM/GRU use PyTorch's native
GPU RNN implementation.

The older drop-in BatchNorm module is still kept in `workarounds/batchnorm_gfx1010.py` for
manual fallback testing:

```python
from workarounds.batchnorm_gfx1010 import BatchNorm2dGFX1010 as BatchNorm2d
# Use exactly like nn.BatchNorm2d -- same constructor args, same behavior
```

## Automatic runtime patching

`install.sh` drops a small `.pth` startup hook into the target venv's `site-packages`.
That hook waits for `torch` to be imported, then automatically imports `workarounds` to
patch the known gfx1010 failure paths:

- `torch.nonzero`
- `torch.masked_select`
- boolean tensor indexing
- tensor `repr`
- `torch.unique`
- unsupported `torch.scatter_add` / `Tensor.scatter_add[_]` layouts

The native `scatter_add` patch routes the common PyG `dim=0` aggregation shape
through compiled GPU `index_add_`, which works on gfx1010. The Python workaround
still keeps a CPU fallback for unsupported scatter layouts.

## Build instructions

```bash
# 1. Clone PyTorch v2.9.1
mkdir ~/pytorch-build && cd ~/pytorch-build
git clone --depth 1 --branch v2.9.1 https://github.com/pytorch/pytorch
git -C pytorch submodule update --init --recursive

# 2. Apply patches
patch -p1 -d pytorch/third_party/composable_kernel < patches/composable_kernel-gfx1010.patch
patch -p1 -d pytorch < patches/pytorch-scatter-add-gfx1010.patch
patch -p1 -d pytorch < patches/pytorch-bn-rnn-gfx1010.patch

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

### PyTorch scatter_add -- `aten/src/ATen/native/cuda/ScatterGatherKernel.cu`

The native PyTorch patch keeps the workaround in compiled code instead of the Python startup
hook. On ROCm gfx1010 only, the common PyG `dim=0` scatter-add shape is routed through
`index_add_`, which runs on the GPU and avoids the failing generic HIP scatter kernel.
Unsupported scatter layouts fall through to PyTorch's existing implementation. CUDA builds and
other ROCm GPU architectures are unchanged.

### PyTorch BatchNorm and RNN dispatch -- `aten/src/ATen/native/Normalization.cpp`, `aten/src/ATen/native/RNN.cpp`

On ROCm gfx1010 only, BatchNorm backend selection skips MIOpen and uses PyTorch's native GPU
BatchNorm kernels. RNN dispatch also skips MIOpen for LSTM/GRU/RNN so PyTorch's native GPU
implementation runs with `torch.backends.cudnn.enabled` left enabled. Other ROCm GPU
architectures are unchanged.
