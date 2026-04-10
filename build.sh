#!/bin/bash
set -e

# Build PyTorch v2.9.1 for AMD gfx1010 (RX 5700 XT / RDNA1)
# Requirements: ROCm 6.4, Python 3.12, ~125 GB free disk
# Edit VENV and SRC to match your paths.

VENV=${VENV:-$HOME/pytorch-gfx1010-venv}
SRC=${SRC:-$HOME/pytorch-build/pytorch}
LOG=${LOG:-$HOME/pytorch-build/build.log}

exec > >(tee -a "$LOG") 2>&1
echo "=== PyTorch gfx1010 build started: $(date) ==="

export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export ROCM_HOME=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx1010
export USE_ROCM=1
export USE_CUDA=0
export USE_MPS=0
export BUILD_TEST=0
export USE_DISTRIBUTED=0
export USE_NCCL=0
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export CMAKE_PREFIX_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH

source "$VENV/bin/activate"
cd "$SRC"

pip install -r requirements.txt

python setup.py clean 2>/dev/null || true
python tools/amd_build/build_amd.py

python setup.py bdist_wheel
WHEEL=$(ls dist/torch-*.whl | head -1)
echo "=== Built wheel: $WHEEL ==="
pip install "$WHEEL" --force-reinstall
echo "=== Install complete: $(date) ==="
python -c "import torch; print('torch:', torch.__version__); print('HIP:', torch.version.hip); print('archs:', torch.cuda.get_arch_list())"
