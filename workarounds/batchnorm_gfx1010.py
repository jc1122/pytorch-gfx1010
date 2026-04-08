"""
BatchNorm2d workaround for AMD gfx1010 (RX 5700 XT / RDNA1).

MIOpen's BatchNorm *backward* kernel uses DPP `v_add_f32 ... row_bcast:15/31`,
GCN-era subgroup broadcast instructions that were re-encoded in RDNA1 and are
not valid on gfx1010. The forward pass works fine; only gradient computation
through `nn.BatchNorm2d` fails with `miopenStatusUnknownError`.

This module provides a drop-in replacement that implements BatchNorm entirely
with element-wise ops and reductions — all of which work on gfx1010.

Usage:
    from workarounds.batchnorm_gfx1010 import BatchNorm2dGFX1010 as BatchNorm2d
    # then use exactly like nn.BatchNorm2d
"""

import torch
import torch.nn as nn


class BatchNorm2dGFX1010(nn.Module):
    """Drop-in replacement for nn.BatchNorm2d on AMD gfx1010 (RDNA1).

    Avoids MIOpen's broken BN backward kernel by computing BatchNorm
    using only element-wise ops and reductions, which all work on gfx1010.

    API is identical to nn.BatchNorm2d.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.lerp_(mean.detach(), self.momentum)
                    self.running_var.lerp_(var.detach(), self.momentum)
                    self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        s = mean[None, :, None, None]
        v = var[None, :, None, None]
        xn = (x - s) / (v + self.eps).sqrt()

        if self.affine:
            xn = xn * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return xn

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )
