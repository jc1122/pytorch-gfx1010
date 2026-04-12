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

This is not a hardware limitation -- all required arithmetic works on gfx1010. All known
failure paths are handled by native PyTorch patches compiled into the wheel.

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
| nonzero / masked_select / bool indexing / tensor repr | NATIVE PATCH | gfx1010 uses an ordered native GPU compaction path instead of broken hipcub `DeviceSelect::Flagged` |
| unique / unique_consecutive | NATIVE PATCH | gfx1010 uses CUDA/HIP sort plus native nonzero compaction; avoids broken hipcub unique/run-length encode and prefix scan |
| scatter_add / PyG sum+mean aggregation | NATIVE PATCH | All layouts work natively; common `dim=0` path is also optimised via compiled GPU `index_add_` |
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
All known gfx1010 failure paths are now handled by native PyTorch patches compiled into
the wheel — there is nothing left to monkey-patch at import time.

`nonzero`, `where(mask)`, `masked_select`, CUDA boolean indexing, and tensor repr use the
native nonzero compaction patch. `unique` and `unique_consecutive` (including `dim=...`)
use the native unique patch. `scatter_add` in all layouts works via the native CUDA kernel;
the common PyG `dim=0` path is also optimised through `index_add_`. `BatchNorm2d` and
`LSTM`/`GRU` skip broken MIOpen paths via native dispatch patches.

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
patch -p1 -d pytorch < patches/pytorch-nonzero-gfx1010.patch
patch -p1 -d pytorch < patches/pytorch-unique-gfx1010.patch

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

PyTorch's `scatter_add_cuda_kernel` uses `gpuAtomicAdd` directly and never touches
rocprim/hipcub device algorithms, so all scatter layouts work on gfx1010 without any
workaround. The patch is a performance optimisation: on ROCm gfx1010 only, the common PyG
`dim=0` scatter-add shape (contiguous broadcast index, matching trailing dimensions) is
routed through `index_add_`. Other shapes, CUDA builds, and other ROCm architectures are
unchanged.

### PyTorch BatchNorm and RNN dispatch -- `aten/src/ATen/native/Normalization.cpp`, `aten/src/ATen/native/RNN.cpp`

On ROCm gfx1010 only, BatchNorm backend selection skips MIOpen and uses PyTorch's native GPU
BatchNorm kernels. RNN dispatch also skips MIOpen for LSTM/GRU/RNN so PyTorch's native GPU
implementation runs with `torch.backends.cudnn.enabled` left enabled. Other ROCm GPU
architectures are unchanged.

### PyTorch nonzero compaction -- `aten/src/ATen/native/cuda/Nonzero.cu`

On ROCm gfx1010 only, dynamic `nonzero` first counts nonzero elements with a small native
kernel, then writes indices through PyTorch's existing ordered block-scan `flag_kernel`.
This avoids the broken hipcub `DeviceSelect::Flagged` path while preserving PyTorch's index
ordering. `torch.where(mask)`, `masked_select`, CUDA boolean indexing, and tensor repr now
run without Python CPU fallbacks.

### PyTorch unique -- `aten/src/ATen/native/cuda/UniqueCub.cu`, `aten/src/ATen/native/cuda/Unique.cu`

On ROCm gfx1010 only, 1-D `unique` and `unique_consecutive` avoid the broken hipcub
unique/run-length encode and prefix-scan path. Non-consecutive unique still uses native
CUDA/HIP sort, then compacts first-of-run positions with the native nonzero patch, computes
inverse indices by binary searching the unique-start positions, and computes counts with a
small kernel. This keeps results on the GPU without the old Python CPU fallback.

For `unique(dim=...)` and `unique_consecutive(dim=...)`, the patch keeps PyTorch's existing
row sort/order logic, then uses row-compare flags plus native nonzero compaction to select
unique rows. Other ROCm GPU architectures and CUDA builds are unchanged.
