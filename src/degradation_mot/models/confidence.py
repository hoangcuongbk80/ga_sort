from __future__ import annotations

from functools import lru_cache
from math import log

import numpy as np
from scipy.special import betaln, betaincc, logsumexp, roots_legendre


@lru_cache(maxsize=16)
def tail_quadrature(threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log weights for integration over the retained score range."""
    nodes, weights = roots_legendre(96)
    half_width = (1.0 - threshold) / 2.0
    return threshold + half_width * (1.0 + nodes), np.log(half_width * weights)


def beta_log_tail(threshold: float, alpha: float, beta: float) -> float:
    if threshold == 0.0:
        return 0.0
    mass = float(betaincc(alpha, beta, threshold))
    if mass > 0.0:
        return log(mass)
    nodes, log_weights = tail_quadrature(threshold)
    return float(logsumexp(
        log_weights + (alpha - 1.0) * np.log(nodes)
        + (beta - 1.0) * np.log1p(-nodes)
    ) - betaln(alpha, beta))
