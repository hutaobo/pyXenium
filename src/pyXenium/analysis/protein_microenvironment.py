from __future__ import annotations

from pyXenium._compat import deprecated_symbol

_PUBLIC_NAMES = {"ProteinMicroEnv"}


def __getattr__(name: str):
    return deprecated_symbol(
        name,
        target_module="pyXenium.multimodal.analysis.protein_microenvironment",
        public_names=_PUBLIC_NAMES,
        old_namespace=__name__,
        new_namespace="pyXenium.multimodal",
    )


__all__ = ["ProteinMicroEnv"]
