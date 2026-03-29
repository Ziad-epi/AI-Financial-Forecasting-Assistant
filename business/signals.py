"""Signal generation from model predictions."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def generate_signal(predicted_return: float, threshold: float = 0.001) -> str:
    """
    Generate a trading-like signal from a predicted return.

    Args:
        predicted_return: Expected return from the forecasting model.
        threshold: Decision threshold around zero for BUY/SELL.

    Returns:
        One of "BUY", "SELL", or "HOLD".
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    if predicted_return > threshold:
        signal = "BUY"
    elif predicted_return < -threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    logger.debug(
        "generate_signal(predicted_return=%s, threshold=%s) -> %s",
        predicted_return,
        threshold,
        signal,
    )
    return signal
