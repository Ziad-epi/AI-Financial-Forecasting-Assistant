"""Integration example for business logic layer."""

from __future__ import annotations

import json
import logging

from business.explain import explain_prediction
from business.risk import compute_risk_level, compute_volatility
from business.signals import generate_signal


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Example model output
    prediction = 0.0021
    returns = [0.011, 0.018, -0.004, 0.012, 0.003]

    signal = generate_signal(prediction, threshold=0.001)
    volatility = compute_volatility(returns)
    risk = compute_risk_level(volatility, low_threshold=0.01, high_threshold=0.02)

    features = {
        "momentum": 0.3,
        "trend": 0.2,
    }
    explanation = explain_prediction(features, prediction)

    output = {
        "prediction": prediction,
        "signal": signal,
        "risk": risk,
        "explanation": explanation,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
