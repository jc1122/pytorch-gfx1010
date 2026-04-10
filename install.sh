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
    echo "WARNING: rocBLAS gfx1010 kernels not found at $TENSILE_YAML"
    echo "  matmul (Linear, BMM, etc.) will not work without them."
    echo "  Install from: https://github.com/jc1122/rocblas-gfx1010"
    echo ""
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

# ── Install workarounds ───────────────────────────────────────────────────────

echo "Installing pytorch-gfx1010-workarounds ..."
pip install "git+$REPO_URL"

# ── Verify ────────────────────────────────────────────────────────────────────

echo ""
echo "=== Running smoke tests ==="
python - <<'PYEOF'
import workarounds
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
check("matmul f32",       lambda: torch.matmul(torch.randn(64,64,device='cuda'), torch.randn(64,64,device='cuda')))
check("nonzero",          lambda: torch.nonzero(torch.tensor([0.,1.,2.],device='cuda')))
check("masked_select",    lambda: torch.masked_select(torch.tensor([1.,-1.],device='cuda'), torch.tensor([True,False],device='cuda')))
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
echo "Then add to your scripts:  import workarounds"
