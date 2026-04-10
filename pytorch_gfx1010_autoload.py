"""Auto-apply gfx1010 workarounds when torch is imported."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys

_APPLIED = False
_FINDER = None


def _apply_workarounds() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    importlib.import_module("workarounds")


class _TorchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        if hasattr(self._wrapped_loader, "create_module"):
            return self._wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module) -> None:
        self._wrapped_loader.exec_module(module)
        _apply_workarounds()


class _TorchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch" or _APPLIED:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _TorchLoader(spec.loader)
        return spec


def install() -> None:
    global _FINDER
    if _APPLIED:
        return
    if "torch" in sys.modules:
        _apply_workarounds()
        return
    _FINDER = _TorchFinder()
    sys.meta_path.insert(0, _FINDER)


install()
