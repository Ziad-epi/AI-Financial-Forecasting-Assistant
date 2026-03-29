"""Risk evaluation utilities."""

from __future__ import annotations

import logging
import math
from typing import Iterable, List

logger = logging.getLogger(__name__)


def _to_float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


def compute_volatility(returns: Iterable[float]) -> float:
    """
    Compute volatility as the standard deviation of returns.

    Args:
        returns: Iterable of periodic returns.

    Returns:
        Volatility (standard deviation). Returns 0.0 for fewer than 2 data points.
    """
    returns_list = _to_float_list(returns)
    n = len(returns_list)
    if n < 2:
        return 0.0

    mean = sum(returns_list) / n
    variance = sum((r - mean) ** 2 for r in returns_list) / (n - 1)
    volatility = math.sqrt(variance)

    logger.debug("compute_volatility(n=%s) -> %s", n, volatility)
    return volatility


def compute_risk_level(
    volatility: float,
    low_threshold: float = 0.01,
    high_threshold: float = 0.02,
) -> str:
    """
    Map volatility to a qualitative risk level.

    Args:
        volatility: Volatility measure (standard deviation).
        low_threshold: Upper bound for LOW RISK.
        high_threshold: Upper bound for MEDIUM RISK.

    Returns:
        One of "LOW RISK", "MEDIUM RISK", or "HIGH RISK".
    """
    if low_threshold <= 0 or high_threshold <= 0:
        raise ValueError("thresholds must be positive")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be less than high_threshold")

    if volatility < low_threshold:
        risk = "LOW RISK"
    elif volatility < high_threshold:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"

    logger.debug(
        "compute_risk_level(volatility=%s, low_threshold=%s, high_threshold=%s) -> %s",
        volatility,
        low_threshold,
        high_threshold,
        risk,
    )
    return risk
