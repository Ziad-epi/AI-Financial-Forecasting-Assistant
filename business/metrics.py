"""Performance and risk metrics for model outputs."""

from __future__ import annotations

import logging
import math
from typing import Iterable, List

logger = logging.getLogger(__name__)


def _to_float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


def calculate_drawdown(prices: Iterable[float]) -> float:
    """
    Calculate the maximum drawdown from a price series.

    Args:
        prices: Iterable of price levels.

    Returns:
        Maximum drawdown as a negative fraction (e.g., -0.12 for -12%).
        Returns 0.0 if fewer than 2 prices are provided.
    """
    prices_list = _to_float_list(prices)
    if len(prices_list) < 2:
        return 0.0

    peak = prices_list[0]
    max_drawdown = 0.0
    for price in prices_list[1:]:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    logger.debug("calculate_drawdown(n=%s) -> %s", len(prices_list), max_drawdown)
    return max_drawdown


def calculate_sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the (simple) Sharpe ratio.

    Args:
        returns: Iterable of periodic returns.
        risk_free_rate: Risk-free rate per period to subtract from returns.

    Returns:
        Sharpe ratio. Returns 0.0 if fewer than 2 returns or zero volatility.
    """
    returns_list = _to_float_list(returns)
    n = len(returns_list)
    if n < 2:
        return 0.0

    adj_returns = [r - risk_free_rate for r in returns_list]
    mean = sum(adj_returns) / n
    variance = sum((r - mean) ** 2 for r in adj_returns) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0

    sharpe = mean / std
    logger.debug("calculate_sharpe_ratio(n=%s) -> %s", n, sharpe)
    return sharpe
