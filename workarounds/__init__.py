"""
workarounds/__init__.py

Import this module to automatically apply all gfx1010 workarounds:

    import workarounds  # patches nn.BatchNorm2d globally

Currently patches:
- torch.nn.BatchNorm2d  -> BatchNorm2dGFX1010
  (MIOpen BN backward uses DPP row_bcast:15/31, not valid on gfx1010 RDNA1)
"""
import torch.nn as nn
from .batchnorm_gfx1010 import BatchNorm2dGFX1010

nn.BatchNorm2d = BatchNorm2dGFX1010
