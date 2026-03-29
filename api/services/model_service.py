from __future__ import annotations

import logging
import math
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from business.explain import explain_prediction
from business.risk import compute_risk_level, compute_volatility
from business.signals import generate_signal

try:
    import joblib
except Exception:  # pragma: no cover - joblib optional
    joblib = None

logger = logging.getLogger(__name__)

FEATURE_ORDER: List[str] = ["lag_1", "lag_2", "lag_3", "MA7", "MA30", "volatility"]


@dataclass
class ModelMetadata:
    model_type: str
    model_path: str
    features: Optional[List[str]]
    n_features: Optional[int]
    loaded_at: str


class ModelService:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.metadata: Optional[ModelMetadata] = None

    def load(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info("Loading model from %s", self.model_path)
        if joblib is not None:
            self.model = joblib.load(self.model_path)
        else:
            with self.model_path.open("rb") as f:
                self.model = pickle.load(f)

        features = None
        n_features = None
        if hasattr(self.model, "feature_names_in_"):
            try:
                features = list(self.model.feature_names_in_)
            except Exception:
                features = None
        if hasattr(self.model, "n_features_in_"):
            try:
                n_features = int(self.model.n_features_in_)
            except Exception:
                n_features = None

        self.metadata = ModelMetadata(
            model_type=self.model.__class__.__name__,
            model_path=str(self.model_path),
            features=features,
            n_features=n_features,
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )

    def _validate_features(self, features: dict) -> None:
        missing = [name for name in FEATURE_ORDER if name not in features]
        if missing:
            raise KeyError(f"Missing feature(s): {', '.join(missing)}")

        for name in FEATURE_ORDER:
            value = features.get(name)
            if value is None:
                raise ValueError(f"Feature '{name}' is None")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{name}' is not numeric") from exc
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                raise ValueError(f"Feature '{name}' is not a finite number")

    def predict(self, features: dict) -> float:
        self.load()
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        self._validate_features(features)
        values = [float(features[name]) for name in FEATURE_ORDER]
        payload = [values]

        try:
            import pandas as pd

            payload = pd.DataFrame(payload, columns=FEATURE_ORDER)
        except Exception:
            pass

        prediction = self.model.predict(payload)
        return float(prediction[0])

    def decision(
        self,
        features: dict,
        signal_threshold: float = 0.001,
        low_vol_threshold: float = 0.01,
        high_vol_threshold: float = 0.02,
    ) -> dict:
        """
        Build a business decision layer on top of model predictions.

        Returns a dict with prediction, signal, risk, and explanation.
        """
        prediction = self.predict(features)
        signal = generate_signal(prediction, threshold=signal_threshold)

        lag_returns = []
        for key in ("lag_1", "lag_2", "lag_3"):
            value = features.get(key)
            if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                lag_returns.append(float(value))

        if len(lag_returns) >= 2:
            volatility = compute_volatility(lag_returns)
        else:
            volatility = float(features.get("volatility", 0.0))

        risk_label = compute_risk_level(
            volatility,
            low_threshold=low_vol_threshold,
            high_threshold=high_vol_threshold,
        )
        risk = risk_label.replace(" RISK", "")

        explanation = explain_prediction(features, prediction)

        logger.info("Decision prediction: %s", prediction)
        logger.info("Decision signal: %s", signal)

        return {
            "prediction": prediction,
            "signal": signal,
            "risk": risk,
            "explanation": explanation,
        }

    def info(self) -> ModelMetadata:
        self.load()
        if self.metadata is None:
            raise RuntimeError("Model metadata not available")
        return self.metadata


def build_model_service() -> ModelService:
    root = Path(__file__).resolve().parents[2]
    model_path = os.getenv("MODEL_PATH", str(root / "model" / "forecast_model.pkl"))
    return ModelService(model_path)
