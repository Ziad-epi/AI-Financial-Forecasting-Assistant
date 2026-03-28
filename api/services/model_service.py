from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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

    def predict(self, features: dict) -> float:
        self.load()
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        values = [features[name] for name in FEATURE_ORDER]
        payload = [values]

        try:
            import pandas as pd

            payload = pd.DataFrame(payload, columns=FEATURE_ORDER)
        except Exception:
            pass

        prediction = self.model.predict(payload)
        return float(prediction[0])

    def info(self) -> ModelMetadata:
        self.load()
        if self.metadata is None:
            raise RuntimeError("Model metadata not available")
        return self.metadata


def build_model_service() -> ModelService:
    root = Path(__file__).resolve().parents[2]
    model_path = os.getenv("MODEL_PATH", str(root / "model" / "forecast_model.pkl"))
    return ModelService(model_path)
