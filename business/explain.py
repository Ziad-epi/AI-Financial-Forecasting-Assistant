"""Heuristic explanations for model predictions."""

from __future__ import annotations

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def explain_prediction(features: Dict[str, Any], prediction: float) -> str:
    """
    Create a simple heuristic explanation for a prediction.

    Args:
        features: Feature dictionary used by the model.
        prediction: Model prediction (e.g., expected return).

    Returns:
        Human-readable explanation string.
    """
    parts: List[str] = []

    # Directional hint from prediction
    if prediction > 0:
        parts.append("upward momentum detected")
    elif prediction < 0:
        parts.append("downward pressure detected")
    else:
        parts.append("signal near neutral")

    # Optional trend signal
    trend = features.get("trend")
    if isinstance(trend, (int, float)):
        if trend > 0:
            parts.append("recent upward trend")
        elif trend < 0:
            parts.append("recent downward trend")

    # Optional volatility signal
    volatility = features.get("volatility")
    low_vol_threshold = 0.01
    high_vol_threshold = 0.02
    if isinstance(volatility, (int, float)):
        if volatility < low_vol_threshold:
            parts.append("low volatility")
        elif volatility > high_vol_threshold:
            parts.append("high volatility")

    # Optional momentum signal
    momentum = features.get("momentum")
    if isinstance(momentum, (int, float)):
        if momentum > 0:
            parts.append("positive momentum")
        elif momentum < 0:
            parts.append("negative momentum")

    explanation = "Prediction is influenced by " + " and ".join(parts)
    logger.debug("explain_prediction(prediction=%s) -> %s", prediction, explanation)
    return explanation
