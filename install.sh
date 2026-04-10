#!/bin/bash
# install.sh — Install PyTorch 2.9.1 + workarounds for AMD RX 5700 XT (gfx1010 / RDNA1)
#
# Requirements:
#   - ROCm 6.4 installed at /opt/rocm
#   - Python 3.12
#   - rocBLAS gfx1010 kernels installed (see github.com/jc1122/rocblas-gfx1010)
#
# Usage:
#   bash install.sh
#   # or with a custom venv path:
#   VENV=~/.venv/my-env bash install.sh

set -e

VENV=${VENV:-$HOME/.venv/gfx1010-pytorch}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
WHEEL_URL="https://github.com/jc1122/pytorch-gfx1010/releases/download/v2.9.1/torch-2.9.1a0+gitd38164a-cp312-cp312-linux_x86_64.whl"
REPO_URL="https://github.com/jc1122/pytorch-gfx1010.git"
ROCBLAS_INSTALL_URL=${ROCBLAS_INSTALL_URL:-https://raw.githubusercontent.com/jc1122/rocblas-gfx1010/main/install.sh}
ROCBLAS_INSTALL_MODE=${ROCBLAS_INSTALL_MODE:-release}
ROCBLAS_ASSET_URL=${ROCBLAS_ASSET_URL:-}
ROCBLAS_SHA256_URL=${ROCBLAS_SHA256_URL:-}

# ── Preflight checks ──────────────────────────────────────────────────────────

echo "=== PyTorch gfx1010 install ==="

if [ ! -f "$ROCM_PATH/bin/hipcc" ]; then
    echo "ERROR: ROCm not found at $ROCM_PATH (set ROCM_PATH if installed elsewhere)"
    exit 1
fi

if ! python3 -c "import sys; assert sys.version_info >= (3,12), 'Python 3.12+ required'" 2>/dev/null; then
    echo "ERROR: Python 3.12+ required (found $(python3 --version 2>&1))"
    exit 1
fi

# Check rocBLAS gfx1010 kernels
TENSILE_YAML="$ROCM_PATH/lib/rocblas/library/TensileLibrary_lazy_gfx1010.yaml"
if [ ! -f "$TENSILE_YAML" ]; then
    echo ""
    echo "rocBLAS gfx1010 kernels not found at $TENSILE_YAML"
    echo "Installing rocBLAS gfx1010 runtime ..."
    echo ""

    ROCBLAS_SCRIPT=$(mktemp)
    trap 'rm -f "$ROCBLAS_SCRIPT"' EXIT
    curl -fsSL "$ROCBLAS_INSTALL_URL" -o "$ROCBLAS_SCRIPT"
    chmod +x "$ROCBLAS_SCRIPT"

    if [ -n "$ROCBLAS_ASSET_URL" ] && [ -n "$ROCBLAS_SHA256_URL" ]; then
        ROCM_PATH="$ROCM_PATH" \
        MODE="$ROCBLAS_INSTALL_MODE" \
        ASSET_URL="$ROCBLAS_ASSET_URL" \
        SHA256_URL="$ROCBLAS_SHA256_URL" \
        "$ROCBLAS_SCRIPT"
    elif [ -n "$ROCBLAS_ASSET_URL" ]; then
        ROCM_PATH="$ROCM_PATH" \
        MODE="$ROCBLAS_INSTALL_MODE" \
        ASSET_URL="$ROCBLAS_ASSET_URL" \
        "$ROCBLAS_SCRIPT"
    else
        ROCM_PATH="$ROCM_PATH" \
        MODE="$ROCBLAS_INSTALL_MODE" \
        "$ROCBLAS_SCRIPT"
    fi

    if [ ! -f "$TENSILE_YAML" ]; then
        echo "ERROR: rocBLAS gfx1010 install did not produce $TENSILE_YAML"
        exit 1
    fi
fi

# ── Create venv ───────────────────────────────────────────────────────────────

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv --without-pip "$VENV"
    curl -sSL https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python"
fi

source "$VENV/bin/activate"

# ── Install PyTorch wheel ─────────────────────────────────────────────────────

echo "Installing PyTorch 2.9.1 (gfx1010 wheel) ..."
pip install --no-deps "$WHEEL_URL"

echo "Installing Python runtime dependencies ..."
pip install \
    "setuptools" \
    "typing_extensions" \
    "filelock" \
    "sympy" \
    "networkx" \
    "jinja2" \
    "fsspec" \
    "pyyaml" \
    "numpy<2"

# ── Install workarounds ───────────────────────────────────────────────────────

echo "Installing pytorch-gfx1010-workarounds ..."
pip install "git+$REPO_URL"

# ── Enable autoload in this venv ──────────────────────────────────────────────

AUTOLOAD_PTH=$("$VENV/bin/python" - <<'PYEOF'
import sysconfig
print(sysconfig.get_path("purelib"))
PYEOF
)
AUTOLOAD_PTH="$AUTOLOAD_PTH/pytorch_gfx1010_autoload.pth"
echo "import pytorch_gfx1010_autoload" > "$AUTOLOAD_PTH"
echo "Installed autoload hook: $AUTOLOAD_PTH"

# ── Verify ────────────────────────────────────────────────────────────────────

echo ""
echo "=== Running smoke tests ==="
python - <<'PYEOF'
import torch, torch.nn as nn, sys

ok = True
def check(name, fn):
    global ok
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        ok = False

check("GPU visible",      lambda: (assert_gpu := torch.cuda.is_available()) or (_ := (_ for _ in ()).throw(AssertionError("no GPU"))))
check("autoload active",  lambda: (not torch.backends.cudnn.enabled) or (_ := (_ for _ in ()).throw(AssertionError("gfx1010 autoload inactive"))))
check("matmul f32",       lambda: torch.matmul(torch.randn(64,64,device='cuda'), torch.randn(64,64,device='cuda')))
check("nonzero",          lambda: torch.nonzero(torch.tensor([0.,1.,2.],device='cuda')))
check("masked_select",    lambda: torch.masked_select(torch.tensor([1.,-1.],device='cuda'), torch.tensor([True,False],device='cuda')))
check("boolean index",    lambda: (t:=torch.randn(2,4,device='cuda'), m:=torch.tensor([True,False,True,False],device='cuda'), t[:, m]))
check("tensor repr",      lambda: repr(torch.tensor([1.,2.],device='cuda')))
check("BatchNorm2d bwd",  lambda: (bn:=nn.BatchNorm2d(4).cuda(), x:=torch.randn(2,4,8,8,device='cuda',requires_grad=True), bn(x).sum().backward()))

if ok:
    print("\nAll checks passed.")
    sys.exit(0)
else:
    print("\nSome checks FAILED.")
    sys.exit(1)
PYEOF

echo ""
echo "=== Install complete ==="
echo "Activate with:  source $VENV/bin/activate"
echo "PyTorch gfx1010 workarounds autoload on first import of torch in this venv."
