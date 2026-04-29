from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any, Iterable

HISTOSEG_ROOT_ENV = "HISTOSEG_ROOT"


def load_histoseg_module(
    *,
    required: Iterable[str],
    histoseg_root: str | Path | None = None,
    purpose: str,
) -> Any:
    return load_histoseg_import(
        "histoseg",
        required=tuple(required),
        histoseg_root=histoseg_root,
        purpose=purpose,
    )


def load_histoseg_submodule(
    module_name: str,
    *,
    required: Iterable[str],
    histoseg_root: str | Path | None = None,
    purpose: str,
) -> Any:
    return load_histoseg_import(
        module_name,
        required=tuple(required),
        histoseg_root=histoseg_root,
        purpose=purpose,
    )


def load_histoseg_import(
    module_name: str,
    *,
    required: tuple[str, ...],
    histoseg_root: str | Path | None,
    purpose: str,
) -> Any:
    try:
        return _import_and_validate(module_name, required=required)
    except Exception as installed_exc:
        roots = _candidate_roots(histoseg_root)
        fallback_errors: list[str] = []
        for root in roots:
            try:
                return _import_from_checkout(module_name, required=required, histoseg_root=root)
            except Exception as fallback_exc:
                fallback_errors.append(f"{root!s}: {fallback_exc}")
        message = (
            f"HistoSeg is required for `{purpose}`. Install or upgrade `histoseg`, "
            "pass `histoseg_root` pointing to a local checkout, or set "
            f"`{HISTOSEG_ROOT_ENV}`."
        )
        if fallback_errors:
            message += " Local checkout import attempts failed: " + " | ".join(fallback_errors)
        raise ImportError(message) from installed_exc


def _candidate_roots(histoseg_root: str | Path | None) -> list[str | Path]:
    roots: list[str | Path] = []
    if histoseg_root is not None:
        roots.append(histoseg_root)
    env_root = os.environ.get(HISTOSEG_ROOT_ENV)
    if env_root:
        roots.append(env_root)
    return roots


def _import_from_checkout(
    module_name: str,
    *,
    required: tuple[str, ...],
    histoseg_root: str | Path,
) -> Any:
    source_root = _resolve_histoseg_source_root(histoseg_root)
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
    _evict_histoseg_modules()
    importlib.invalidate_caches()
    return _import_and_validate(module_name, required=required)


def _resolve_histoseg_source_root(histoseg_root: str | Path) -> Path:
    root = Path(histoseg_root).expanduser().resolve()
    src_root = root / "src"
    if src_root.exists():
        return src_root
    if root.name == "src" and root.exists():
        return root
    if (root / "histoseg").exists():
        return root
    raise ImportError(f"Could not locate a HistoSeg source package under {root!s}.")


def _evict_histoseg_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "histoseg" or module_name.startswith("histoseg."):
            sys.modules.pop(module_name, None)


def _import_and_validate(module_name: str, *, required: tuple[str, ...]) -> Any:
    module = importlib.import_module(module_name)
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"The HistoSeg module `{module_name}` is missing required API: "
            f"{', '.join(missing)}"
        )
    return module
