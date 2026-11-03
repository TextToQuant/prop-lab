"""Statistical significance of a strategy's edge.

Why this exists
---------------
A walk-forward verdict of ``holds`` says the edge did not *degrade*. It says
nothing about whether there *was* an edge. A strategy whose out-of-sample
mean R is +0.009 can earn ``holds`` while its bootstrap confidence interval
spans zero, in which case the honest statement is "no edge was detected",
not "the edge held".

This module answers the question the verdict cannot: is the observed mean R
distinguishable from zero?

Method
------
The primary estimator is a **bootstrap confidence interval on the mean R**,
computed by resampling the realised trade R-multiples with replacement and
taking percentile bounds of the resampled means. The bootstrap is used
rather than the normal approximation because the distribution of per-trade
R is not normal: it is bounded below at -1 (a stopped-out trade cannot lose
more than one R) and heavily skewed by partial exits.

A t-statistic is reported alongside as a cross-check with a different
assumption. When the two disagree, the bootstrap interval is the one the
package trusts, and the disagreement is surfaced in the warning text.

Sample size
-----------
Below :data:`MIN_TRADES_FOR_SIGNIFICANCE` trades the interval is not
trustworthy, and the result is reported as inconclusive regardless of
whether the interval excludes zero. A narrow interval computed from twenty
trades is narrow for the wrong reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EdgeTest",
    "test_edge",
    "MIN_TRADES_FOR_SIGNIFICANCE",
    "DEFAULT_BOOTSTRAP_ATTEMPTS",
]

#: Below this many trades the bootstrap interval is too wide to support a
#: conclusion, and the result is reported as inconclusive.
MIN_TRADES_FOR_SIGNIFICANCE: int = 30

#: Bootstrap resamples used to build the confidence interval.
DEFAULT_BOOTSTRAP_ATTEMPTS: int = 10_000


@dataclass
class EdgeTest:
    """The result of testing whether a trade distribution has a real edge.

    Attributes:
        n_trades: number of trades in the sample.
        mean_r: observed mean R per trade.
        stderr: standard error of the mean R.
        t_statistic: mean R divided by its standard error.
        ci_low: lower bound of the 95 percent confidence interval on mean R.
        ci_high: upper bound of the same interval.
        significant: True when the interval excludes zero AND the sample is
            large enough to support a conclusion.
        conclusive: True when the sample is large enough to support a
            conclusion at all. When False, ``significant`` is False even if
            the interval happens to exclude zero.
        method: the estimator used, ``'bootstrap'``.
        n_attempts: bootstrap resamples performed.
        warnings: caveats worth attaching to the result.
    """

    n_trades: int
    mean_r: float
    stderr: float
    t_statistic: float
    ci_low: float
    ci_high: float
    significant: bool
    conclusive: bool
    method: str = "bootstrap"
    n_attempts: int = DEFAULT_BOOTSTRAP_ATTEMPTS
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """One-paragraph plain-text summary of the test."""
        lines = [
            f"trades        : {self.n_trades}",
            f"mean R        : {self.mean_r:+.4f}",
            f"stderr        : {self.stderr:.4f}",
            f"t-statistic   : {self.t_statistic:+.2f}",
            f"95% CI        : [{self.ci_low:+.4f}, {self.ci_high:+.4f}]",
        ]
        if not self.conclusive:
            lines.append(
                "verdict       : inconclusive (too few trades to support a conclusion)"
            )
        elif self.significant:
            direction = "positive" if self.mean_r > 0 else "negative"
            lines.append(
                f"verdict       : the edge is {direction} and distinguishable from zero"
            )
        else:
            lines.append(
                "verdict       : no edge detected (the interval includes zero)"
            )
        return "\n".join(lines)

    def warning(self) -> str:
        """A single-sentence caveat suitable for a warnings list."""
        if not self.conclusive:
            return (
                f"the edge could not be tested: only {self.n_trades} trades "
                f"(need >= {MIN_TRADES_FOR_SIGNIFICANCE}) for a confidence "
                "interval on the mean R to mean anything"
            )
        if self.significant:
            return ""
        return (
            f"no edge was detected: the out-of-sample mean R of {self.mean_r:+.4f} "
            f"has a 95 percent bootstrap interval of [{self.ci_low:+.4f}, "
            f"{self.ci_high:+.4f}], which includes zero. The strategy is not "
            "distinguishable from a random trade sequence at this sample size"
        )


def test_edge(
    pnl_r_array: np.ndarray,
    *,
    n_attempts: int = DEFAULT_BOOTSTRAP_ATTEMPTS,
    seed: int = 42,
    confidence: float = 0.95,
) -> EdgeTest:
    """Test whether a set of per-trade R-multiples has an edge.

    Args:
        pnl_r_array: per-trade ``pnl_r`` values from a real backtest.
        n_attempts: bootstrap resamples. 10,000 is the default because the
            percentile bounds of a bootstrap converge slowly.
        seed: RNG seed, so the result is reproducible.
        confidence: confidence level for the interval.

    Returns:
        An :class:`EdgeTest`. Never raises for an empty input: an empty
        sample returns an inconclusive result with zeroed numbers, because
        "no trades" is a fact the caller needs to see, not an exception.

    Raises:
        ValueError: if ``n_attempts`` is below 100 or ``confidence`` is not
            in (0, 1).
    """
    if n_attempts < 100:
        raise ValueError(f"n_attempts must be at least 100, got {n_attempts!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")

    values = np.asarray(pnl_r_array, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return EdgeTest(
            n_trades=0,
            mean_r=0.0,
            stderr=0.0,
            t_statistic=0.0,
            ci_low=0.0,
            ci_high=0.0,
            significant=False,
            conclusive=False,
            n_attempts=n_attempts,
            warnings=["there were no trades to test"],
        )

    n = int(values.size)
    mean_r = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    t_statistic = float(mean_r / stderr) if stderr > 0 else 0.0

    rng = np.random.default_rng(seed)
    means = np.empty(n_attempts, dtype=float)
    # Resample in blocks to keep the memory cost flat for large samples.
    block = max(1, min(n_attempts, 4096))
    done = 0
    while done < n_attempts:
        take = min(block, n_attempts - done)
        picks = rng.integers(0, n, size=(take, n))
        means[done : done + take] = values[picks].mean(axis=1)
        done += take

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(means, 100.0 * alpha))
    ci_high = float(np.percentile(means, 100.0 * (1.0 - alpha)))

    conclusive = n >= MIN_TRADES_FOR_SIGNIFICANCE
    excludes_zero = ci_low > 0.0 or ci_high < 0.0

    warnings: list[str] = []
    if not conclusive:
        warnings.append(
            f"sample too small to conclude: {n} trades, "
            f"{MIN_TRADES_FOR_SIGNIFICANCE} needed"
        )
    if stderr > 0 and abs(t_statistic) < 2.0 and excludes_zero:
        warnings.append(
            f"the bootstrap interval excludes zero (t = {t_statistic:+.2f}) but "
            "the t-statistic is small; the interval is narrow because the "
            "trade R-multiples are bounded, not because the edge is strong"
        )

    return EdgeTest(
        n_trades=n,
        mean_r=mean_r,
        stderr=stderr,
        t_statistic=t_statistic,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=bool(conclusive and excludes_zero),
        conclusive=conclusive,
        n_attempts=n_attempts,
        warnings=warnings,
    )
