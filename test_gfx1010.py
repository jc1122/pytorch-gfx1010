#!/usr/bin/env python3
"""
Smoke-test suite for PyTorch gfx1010 (AMD RX 5700 XT / RDNA1).

Usage:
    python test_gfx1010.py

Exits with 0 if all tests pass, or the number of failures otherwise.

Prerequisites:
    - PyTorch built for gfx1010 (see build.sh)
    - rocBLAS gfx1010 kernels installed (see github.com/jc1122/rocblas-gfx1010)
    - This package installed: pip install -e .
"""

import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda"


def run(tests):
    passed, failed = 0, []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed.append(fn.__name__)
    return passed, failed


# ── Baseline ──────────────────────────────────────────────────────────────────

def test_gpu_visible():
    assert torch.cuda.is_available(), "no GPU visible"
    name = torch.cuda.get_device_name(0)
    assert "gfx1010" in torch.cuda.get_device_properties(0).gcnArchName or \
           "5700" in name or "Navi" in name, \
        f"unexpected GPU: {name}"


def test_torch_version():
    assert torch.version.hip is not None, "HIP not enabled"


# ── Element-wise and reductions ───────────────────────────────────────────────

def test_elementwise():
    a = torch.randn(128, 128, device=DEVICE)
    b = torch.randn(128, 128, device=DEVICE)
    _ = a + b
    _ = a * b
    _ = a.relu()
    _ = a.sigmoid()
    _ = a.exp()


def test_reductions():
    a = torch.randn(256, 256, device=DEVICE)
    _ = a.sum()
    _ = a.mean()
    _ = a.max()
    _ = a.norm()


# ── matmul / BLAS (requires rocblas-gfx1010) ─────────────────────────────────

def test_matmul_float32():
    a = torch.randn(128, 256, device=DEVICE)
    b = torch.randn(256, 64, device=DEVICE)
    c = torch.matmul(a, b)
    assert c.shape == (128, 64)


def test_matmul_float16():
    a = torch.randn(128, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 64, device=DEVICE, dtype=torch.float16)
    c = torch.matmul(a, b)
    assert c.shape == (128, 64)


def test_matmul_bfloat16():
    a = torch.randn(128, 256, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 64, device=DEVICE, dtype=torch.bfloat16)
    c = torch.matmul(a, b)
    assert c.shape == (128, 64)


def test_batched_matmul():
    a = torch.randn(4, 64, 128, device=DEVICE)
    b = torch.randn(4, 128, 32, device=DEVICE)
    c = torch.bmm(a, b)
    assert c.shape == (4, 64, 32)


# ── nn.Linear ─────────────────────────────────────────────────────────────────

def test_linear_forward_backward():
    m = nn.Linear(128, 64).to(DEVICE)
    x = torch.randn(16, 128, device=DEVICE, requires_grad=True)
    loss = m(x).sum()
    loss.backward()
    assert x.grad is not None
    assert m.weight.grad is not None


# ── Conv2d ────────────────────────────────────────────────────────────────────

def test_conv2d_forward_backward():
    m = nn.Conv2d(8, 16, 3, padding=1).to(DEVICE)
    x = torch.randn(2, 8, 32, 32, device=DEVICE, requires_grad=True)
    loss = m(x).sum()
    loss.backward()
    assert x.grad is not None


# ── Attention ─────────────────────────────────────────────────────────────────

def test_scaled_dot_product_attention():
    q = torch.randn(2, 4, 16, 32, device=DEVICE)
    k = torch.randn(2, 4, 16, 32, device=DEVICE)
    v = torch.randn(2, 4, 16, 32, device=DEVICE)
    out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == q.shape


# ── LayerNorm / GroupNorm ─────────────────────────────────────────────────────

def test_layernorm_backward():
    m = nn.LayerNorm(64).to(DEVICE)
    x = torch.randn(4, 16, 64, device=DEVICE, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None


def test_groupnorm_backward():
    m = nn.GroupNorm(4, 16).to(DEVICE)
    x = torch.randn(2, 16, 8, 8, device=DEVICE, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None


# ── BatchNorm2d (requires workarounds) ───────────────────────────────────────

def test_batchnorm2d_forward_backward():
    import workarounds  # noqa: F401 — applies monkey-patch
    bn = nn.BatchNorm2d(8).to(DEVICE)
    x = torch.randn(2, 8, 16, 16, device=DEVICE, requires_grad=True)
    loss = bn(x).sum()
    loss.backward()
    assert x.grad is not None
    assert bn.weight.grad is not None


def test_batchnorm2d_eval():
    import workarounds  # noqa: F401
    bn = nn.BatchNorm2d(8).to(DEVICE).eval()
    x = torch.randn(2, 8, 16, 16, device=DEVICE)
    out = bn(x)
    assert out.shape == x.shape


# ── Optimizer step ────────────────────────────────────────────────────────────

def test_adamw_step():
    m = nn.Linear(64, 32).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(8, 64, device=DEVICE)
    opt.zero_grad()
    m(x).sum().backward()
    opt.step()


# ── workaround patches (requires workarounds) ─────────────────────────────────

def test_nonzero():
    import workarounds  # noqa: F401
    a = torch.tensor([0., 1., 0., 2.], device=DEVICE)
    idx = torch.nonzero(a)
    assert idx.shape == (2, 1)
    assert idx.device.type == "cuda"


def test_masked_select():
    import workarounds  # noqa: F401
    a = torch.tensor([1., -1., 2., -2.], device=DEVICE)
    mask = torch.tensor([True, False, True, False], device=DEVICE)
    result = torch.masked_select(a, mask)
    assert result.tolist() == [1.0, 2.0]
    assert result.device.type == "cuda"


def test_boolean_indexing():
    import workarounds  # noqa: F401
    # Direct
    a = torch.tensor([1., -1., 2., -2.], device=DEVICE)
    result = a[a > 0]
    assert result.tolist() == [1.0, 2.0]
    assert result.device.type == "cuda"

    # Slicing + Mask (used by torch_geometric edge_index[:, mask])
    b = torch.tensor([[0, 1, 2], [1, 2, 0]], device=DEVICE)
    mask = torch.tensor([True, False, True], device=DEVICE)
    result2 = b[:, mask]
    assert result2.tolist() == [[0, 2], [1, 0]]
    assert result2.device.type == "cuda"



def test_tensor_repr():
    import workarounds  # noqa: F401
    a = torch.tensor([1., 2., 3.], device=DEVICE)
    s = repr(a)
    assert "cuda" in s, f"device not in repr: {s}"


def test_unique():
    import workarounds  # noqa: F401
    a = torch.tensor([3., 1., 2., 1., 3.], device=DEVICE)
    u = torch.unique(a, sorted=True)
    assert u.tolist() == [1.0, 2.0, 3.0]
    assert u.device.type == "cuda"


def test_lstm():
    import workarounds  # noqa: F401
    lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True).to(DEVICE)
    x = torch.randn(4, 10, 16, device=DEVICE)
    out, (h, c) = lstm(x)
    assert out.shape == (4, 10, 32)
    loss = out.sum()
    loss.backward()
    assert lstm.weight_ih_l0.grad is not None


def test_gru():
    import workarounds  # noqa: F401
    gru = nn.GRU(input_size=16, hidden_size=32, num_layers=1, batch_first=True).to(DEVICE)
    x = torch.randn(4, 10, 16, device=DEVICE)
    out, h = gru(x)
    assert out.shape == (4, 10, 32)
    out.sum().backward()
    assert gru.weight_ih_l0.grad is not None


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_gpu_visible,
    test_torch_version,
    test_elementwise,
    test_reductions,
    test_matmul_float32,
    test_matmul_float16,
    test_matmul_bfloat16,
    test_batched_matmul,
    test_linear_forward_backward,
    test_conv2d_forward_backward,
    test_scaled_dot_product_attention,
    test_layernorm_backward,
    test_groupnorm_backward,
    test_batchnorm2d_forward_backward,
    test_batchnorm2d_eval,
    test_adamw_step,
    test_nonzero,
    test_masked_select,
    test_boolean_indexing,
    test_tensor_repr,
    test_unique,
    test_lstm,
    test_gru,
]

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}  HIP {torch.version.hip}")
    if torch.cuda.is_available():
        print(f"GPU     {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU     NOT VISIBLE\n")

    passed, failed = run(ALL_TESTS)
    total = len(ALL_TESTS)
    print(f"\n{passed}/{total} passed", end="")
    if failed:
        print(f"  —  FAILED: {', '.join(failed)}")
    else:
        print("  —  ALL OK")
    sys.exit(len(failed))
