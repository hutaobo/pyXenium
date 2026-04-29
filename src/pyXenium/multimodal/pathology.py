from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

__all__ = ["BMNetBackend", "NamedPathologyFeatureBackend"]


class NamedPathologyFeatureBackend(Protocol):
    """Protocol for H&E backends that emit named contour or zone features."""

    feature_prefix: str

    def extract_features(self, image_patch: Any, **metadata: Any) -> Mapping[str, float]:
        """Return named numeric features for one H&E patch or masked zone."""


@dataclass
class BMNetBackend:
    """Adapter for breast H&E BM-Net style classifiers.

    The adapter intentionally does not bundle model weights or training code. Pass a
    callable or model object that returns class probabilities for one image patch.
    """

    model: Any | None = None
    predict_fn: Callable[..., Any] | None = None
    labels: Sequence[str] = ("normal", "benign", "in_situ", "invasive")
    feature_prefix: str = "bmnet"

    def extract_features(self, image_patch: Any, **metadata: Any) -> dict[str, float]:
        probabilities = self._predict_probabilities(image_patch, **metadata)
        features = {f"{label}_prob": float(probabilities.get(label, 0.0)) for label in self.labels}
        tumor_prob = float(features.get("in_situ_prob", 0.0) + features.get("invasive_prob", 0.0))
        features["tumor_prob"] = tumor_prob
        features["invasive_margin"] = float(
            features.get("invasive_prob", 0.0)
            - max(
                features.get("normal_prob", 0.0),
                features.get("benign_prob", 0.0),
                features.get("in_situ_prob", 0.0),
            )
        )
        features["prediction_entropy"] = _probability_entropy(features[f"{label}_prob"] for label in self.labels)
        features["majority_label_code"] = float(
            np.argmax([features[f"{label}_prob"] for label in self.labels]) if self.labels else -1
        )
        return features

    def _predict_probabilities(self, image_patch: Any, **metadata: Any) -> dict[str, float]:
        predictor = self.predict_fn
        if predictor is None and self.model is not None:
            for name in ("predict_proba", "predict", "__call__"):
                candidate = getattr(self.model, name, None)
                if candidate is not None:
                    predictor = candidate
                    break
        if predictor is None:
            raise ValueError("BMNetBackend requires `predict_fn` or a model with predict/predict_proba.")

        try:
            raw = predictor(image_patch, **metadata)
        except TypeError:
            raw = predictor(image_patch)
        return _coerce_probabilities(raw, labels=self.labels)


def _coerce_probabilities(raw: Any, *, labels: Sequence[str]) -> dict[str, float]:
    if isinstance(raw, Mapping):
        values = {
            (str(key)[: -len("_prob")] if str(key).endswith("_prob") else str(key)): float(value)
            for key, value in raw.items()
        }
    else:
        array = np.asarray(raw, dtype=float).reshape(-1)
        values = {str(label): float(array[index]) for index, label in enumerate(labels[: len(array)])}

    total = float(sum(max(float(values.get(label, 0.0)), 0.0) for label in labels))
    if total > 0:
        return {str(label): max(float(values.get(label, 0.0)), 0.0) / total for label in labels}
    return {str(label): 0.0 for label in labels}


def _probability_entropy(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return 0.0
    array = array / array.sum()
    return float(-np.sum(array * np.log2(array)))
