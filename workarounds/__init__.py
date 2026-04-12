"""
workarounds/__init__.py

All gfx1010 failure paths are now handled by native PyTorch patches compiled
into the wheel.  Nothing needs to be monkey-patched at import time.

The batchnorm_gfx1010 module is kept for manual drop-in use only:

    from workarounds.batchnorm_gfx1010 import BatchNorm2dGFX1010 as BatchNorm2d
"""
