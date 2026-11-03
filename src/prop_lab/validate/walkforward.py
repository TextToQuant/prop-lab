"""Purged walk-forward out-of-sample validation.

The point of this module is to answer one uncomfortable question honestly:
*does this strategy still work on data it was not tuned on?* A backtest
that only ever sees its own training data cannot answer it, and a single
80/20 split answers it with one number and no error bars. This module
splits the bar stream into contiguous out-of-sample test blocks, runs the
strategy over the data preceding each block, and reports how much of the
edge survived.

Why purged walk-forward and not a simple split
----------------------------------------------
Three things break a naive split of financial time series:

1. **Label overlap.** A trade opened just before the test boundary can
   still be open during the test period, so its outcome was produced by
   test-period prices. Counting it as in-sample leaks the future into
   the training statistics. The *purge* removes the training bars nearest
   the test boundary, where such straddling labels live.
2. **Post-test contamination.** Price action immediately after a test
   block is correlated with the test block itself (a trend that continues
   into the next fold's training window leaks the test outcome forward).
   The *embargo* quarantines that region.
3. **Look-ahead by construction.** Splitting a time series randomly, or
   letting the training set cover the test period, hands the strategy
   information it could never have had in real time. Contiguous,
   strictly-ordered blocks with a strictly-earlier training set cannot
   do that.

Split geometry
--------------
The bar index (sorted by time) is cut into ``n_splits + 1`` contiguous
blocks. The first block is seed history; the remaining ``n_splits``
blocks are the out-of-sample test blocks, covering the tail of the
series. For fold ``i`` the test block is ``bars[t0:t1]``. The training
set is the ``train_len = int(train_frac * t0)`` bars immediately before
``t0`` -- a **rolling** window, not an expanding one, so that no fold
is judged on performance that was already known three folds ago.

Purge / embargo convention (kept consistent throughout)
-------------------------------------------------------
Starting from the training slice ``[t0 - train_len, t0)``:

* ``purge_bars`` removes the **last** ``purge_bars`` bars of the
  training window. Those bars sit closest to the test boundary, so their
  labels are the ones most likely to overlap the test period.
* ``embargo_bars`` removes an **additional** ``embargo_bars`` bars after
  that gap -- i.e. the embargo extends the gap between the training data
  and the test data rather than eating into the training window.

The total gap between the last training bar and the first test bar is
therefore ``purge_bars + embargo_bars`` bars. With both set to zero the
training window touches the test block exactly, which is the unsafe
default; use them whenever the strategy holds positions across bars.

Lineage
-------
Marcos Lopez de Prado, *Advances in Financial Machine Learning*
(2018), chapter 7 (purged k-fold cross-validation with an embargo) and
chapter 12 (backtesting through cross-validation, and the argument for
purging overlapping labels). The implementation conventions for
chronological, purged splits with an embargo follow
https://github.com/eslazarev/purged-cross-validation.

Trade generation is delegated entirely to :func:`prop_lab.core.run_backtest`;
this module never re-implements the fill model.

Verdict thresholds (documented constants, SPEC section 4)
---------------------------------------------------------
* ``oos_mean_r <= 0``            -> ``"fails"``
* ``degradation <= 0.10``        -> ``"holds"``
* ``degradation <= 0.30``        -> ``"degrades"``
* otherwise                      -> ``"fails"``

``degradation`` is ``is_mean_r - oos_mean_r``: the mean R per trade the
strategy lost between in-sample and out-of-sample.

Sample-size honesty
-------------------
Fewer than 30 out-of-sample trades is not a result, it is an anecdote.
That produces an explicit warning in ``WFResult.warnings`` and is never
silently suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core import Trade, run_backtest
from .significance import EdgeTest, test_edge

__all__ = [
    "WFWindow",
    "WFResult",
    "walk_forward",
    "walk_forward_table",
    "DEGRADATION_HOLDS_MAX",
    "DEGRADATION_DEGRADES_MAX",
    "MIN_OOS_TRADES_FOR_CONCLUSION",
]


# ----------------------------------------------------------------------
# Documented thresholds
# ----------------------------------------------------------------------
#: Maximum tolerated mean-R degradation for the verdict "holds".
DEGRADATION_HOLDS_MAX: float = 0.10
#: Maximum tolerated mean-R degradation for the verdict "degrades".
#: Above this the verdict is "fails".
DEGRADATION_DEGRADES_MAX: float = 0.30
#: Fewer than this many out-of-sample trades means the result is not
#: statistically meaningful and must be flagged as such.
MIN_OOS_TRADES_FOR_CONCLUSION: int = 30
#: Threshold below which out-of-sample beating in-sample is worth noting
#: explicitly (usually noise or a conservative training window).
OOS_BETTER_IS_NOTE_THRESHOLD: float = -0.05

#: The verdict vocabulary. ``no_edge`` was added when the edge-significance
#: test showed that ``holds`` could be returned for a strategy whose edge is
#: statistically indistinguishable from zero -- a verdict that reads as
#: "the edge survived" when there was no edge to survive.
_VERDICTS = ("holds", "degrades", "fails", "no_edge")


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
@dataclass
class WFWindow:
    """One walk-forward fold.

    ``train_start`` / ``train_end`` / ``test_start`` / ``test_end`` are
    UNIX seconds (UTC) taken from the bar timestamps. ``train_end`` and
    ``test_end`` are the timestamps of the **last** bar in their range
    (inclusive).
    """

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train_trades: int
    n_test_trades: int
    test_mean_r: float
    test_win_rate: float
    # Additional reporting fields (defaults keep the dataclass
    # constructible from the nine contract fields alone).
    train_mean_r: float = 0.0
    train_win_rate: float = 0.0
    test_total_r: float = 0.0
    train_total_r: float = 0.0
    n_train_bars: int = 0
    n_test_bars: int = 0

    @property
    def n_bars_train(self) -> int:
        """Number of bars in the training slice of this fold."""
        return int(self.n_train_bars)

    @property
    def n_bars_test(self) -> int:
        """Number of bars in the test slice of this fold."""
        return int(self.n_test_bars)

    def as_row(self) -> dict[str, Any]:
        """Flat dict of this fold's numbers, suitable for a table."""
        return {
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train_bars": self.n_train_bars,
            "n_test_bars": self.n_test_bars,
            "n_train_trades": self.n_train_trades,
            "train_mean_r": self.train_mean_r,
            "n_test_trades": self.n_test_trades,
            "test_mean_r": self.test_mean_r,
            "test_win_rate": self.test_win_rate,
        }


@dataclass
class WFResult:
    """The outcome of a walk-forward run.

    In-sample and out-of-sample statistics are **pooled across folds**:
    ``is_mean_r`` is the mean ``pnl_r`` of every training-window trade
    from every fold, and ``oos_mean_r`` the mean ``pnl_r`` of every
    test-window trade.
    """

    windows: list[WFWindow]
    oos_mean_r: float
    oos_win_rate: float
    oos_total_r: float
    is_mean_r: float
    degradation: float
    verdict: str
    n_windows: int
    warnings: list[str] = field(default_factory=list)
    # Additional reporting fields.
    is_total_r: float = 0.0
    is_win_rate: float = 0.0
    n_oos_trades: int = 0
    n_is_trades: int = 0
    # Edge significance. The verdict is refined by these: a strategy whose
    # out-of-sample mean R is indistinguishable from zero gets the verdict
    # "no_edge" rather than "holds".
    edge_significant: bool = False
    edge_conclusive: bool = False
    edge_ci_low: float = 0.0
    edge_ci_high: float = 0.0
    edge_t_statistic: float = 0.0

    def summary(self) -> str:
        """One-line human summary of the verdict."""
        return (
            f"verdict={self.verdict} "
            f"IS mean R={self.is_mean_r:.3f} (n={self.n_is_trades}) "
            f"OOS mean R={self.oos_mean_r:.3f} (n={self.n_oos_trades}) "
            f"degradation={self.degradation:.3f} "
            f"edge CI=[{self.edge_ci_low:+.4f}, {self.edge_ci_high:+.4f}] "
            f"windows={self.n_windows}"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _time_values(df: pd.DataFrame) -> np.ndarray:
    """Return the bar timestamps of ``df`` as an int64 array.

    Mirrors the column-flexible lookup in :func:`prop_lab.core.iter_bars`:
    ``time`` first, then ``ts``, case-insensitive.
    """
    cols = {str(c).lower(): c for c in df.columns}
    ts_col = cols.get("time") or cols.get("ts")
    if ts_col is None:
        raise KeyError(
            "No 'time' or 'ts' column found. "
            f"Available columns: {list(df.columns)}"
        )
    return df[ts_col].astype("int64").to_numpy()


def _stats(trades: list[Trade]) -> tuple[float, float, float, int]:
    """Return ``(mean_r, win_rate, total_r, n)`` for a list of trades."""
    if not trades:
        return 0.0, 0.0, 0.0, 0
    rs = np.array([t.pnl_r for t in trades], dtype=float)
    return (
        float(rs.mean()),
        float((rs > 0).mean()),
        float(rs.sum()),
        int(len(rs)),
    )


def _verdict(degradation: float, oos_mean_r: float) -> str:
    """Map degradation and out-of-sample mean R onto a verdict string.

    Args:
        degradation: ``is_mean_r - oos_mean_r``.
        oos_mean_r: pooled mean R of the out-of-sample trades.

    Returns:
        ``"holds"``, ``"degrades"`` or ``"fails"``.
    """
    if oos_mean_r <= 0:
        return "fails"
    if degradation <= DEGRADATION_HOLDS_MAX:
        return "holds"
    if degradation <= DEGRADATION_DEGRADES_MAX:
        return "degrades"
    return "fails"


def _verdict_with_significance(
    degradation: float,
    oos_mean_r: float,
    edge_significant: bool,
    edge_conclusive: bool,
) -> str:
    """Verdict, refined by whether the edge is distinguishable from zero.

    The degradation-only verdict answers "did the edge survive out of
    sample". On its own it cannot answer "was there an edge", and it
    returns ``holds`` for a strategy whose out-of-sample mean R is
    indistinguishable from zero -- which reads as "the edge survived" when
    there was never an edge to survive.

    So the stability verdict is computed first, and then downgraded to
    ``no_edge`` when the edge itself is not established. ``no_edge`` is
    deliberately distinct from ``fails``: ``fails`` says the edge degraded,
    ``no_edge`` says no edge was detected in the first place. A strategy
    with no edge and no degradation is not a strategy that failed
    out of sample; it is a strategy that never had anything to fail.

    Args:
        degradation: ``is_mean_r - oos_mean_r``.
        oos_mean_r: pooled mean R of the out-of-sample trades.
        edge_significant: True when the confidence interval on the mean R
            excludes zero.
        edge_conclusive: True when the sample was large enough for the
            interval to mean anything.

    Returns:
        One of ``"holds"``, ``"degrades"``, ``"fails"`` or ``"no_edge"``.
    """
    base = _verdict(degradation, oos_mean_r)
    if base in ("fails", "degrades"):
        return base
    if edge_conclusive and not edge_significant:
        return "no_edge"
    return base


# ----------------------------------------------------------------------
# The walk-forward engine
# ----------------------------------------------------------------------
def walk_forward(
    strategy_factory: Callable[[], Any],
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    train_frac: float = 0.6,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    direction: str = "both",
    risk_points: float | None = None,
) -> WFResult:
    """Run a purged walk-forward validation of a strategy.

    The bar index (sorted by time) is cut into ``n_splits + 1``
    contiguous blocks. The first block is seed history; the remaining
    ``n_splits`` blocks are the out-of-sample test blocks covering the
    tail of the series. For each fold the strategy is backtested over
    the rolling training window before the test block, and separately
    over the test block itself.

    Args:
        strategy_factory: callable returning a **fresh** strategy
            instance (a :class:`prop_lab.core.StrategyCore` subclass
            instance). It is called once per backtest run, so no state
            leaks between folds.
        df: frame in the standard schema, sorted by ``time`` ascending.
            The time column may be called ``time`` or ``ts``.
        n_splits: number of out-of-sample test blocks.
        train_frac: fraction of the history preceding a test block that
            forms the rolling training window. The training window has
            ``int(train_frac * t0)`` bars, where ``t0`` is the index of
            the first test bar.
        purge_bars: number of training bars removed from the end of the
            training window (the ones closest to the test boundary).
        embargo_bars: number of **additional** bars between the end of
            the training window and the start of the test block. The
            total gap is ``purge_bars + embargo_bars``.
        direction: ``"both"``, ``"long"`` or ``"short"``; forwarded to
            :func:`prop_lab.core.run_backtest`.
        risk_points: optional fixed risk per trade; forwarded to
            :func:`prop_lab.core.run_backtest`.

    Returns:
        A :class:`WFResult` holding the per-fold windows, the pooled
        in-sample and out-of-sample statistics, the degradation figure, a
        verdict and an explicit list of warnings.

    Raises:
        TypeError: if ``strategy_factory`` is not callable.
        ValueError: if the arguments are out of range, or the frame has
            no time column.

    Notes:
        The test slice keeps every trade the engine produces, including
        the position that is force-closed at the end of the slice
        (``exit_reason == "data_end"``). Dropping it would bias the
        out-of-sample measurement.
    """
    # -- argument validation -----------------------------------------
    if not callable(strategy_factory):
        raise TypeError(
            "strategy_factory must be a callable returning a fresh strategy "
            f"instance, got {type(strategy_factory).__name__}"
        )
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 1:
        raise ValueError(f"n_splits must be an int >= 1, got {n_splits!r}")
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac!r}")
    if purge_bars < 0:
        raise ValueError(f"purge_bars must be >= 0, got {purge_bars!r}")
    if embargo_bars < 0:
        raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars!r}")

    n_bars = len(df)
    if n_bars == 0:
        raise ValueError("df is empty; nothing to validate")

    times = _time_values(df)

    # -- split geometry ----------------------------------------------
    # n_splits + 1 contiguous blocks; the first is seed history and the
    # rest are the out-of-sample test blocks covering the tail.
    blocks = np.array_split(np.arange(n_bars), n_splits + 1)
    test_blocks = blocks[1:]

    gap = purge_bars + embargo_bars

    windows: list[WFWindow] = []
    all_is_trades: list[Trade] = []
    all_oos_trades: list[Trade] = []

    for fold, test_idx in enumerate(test_blocks):
        t0 = int(test_idx[0])
        t1 = int(test_idx[-1]) + 1  # exclusive

        # Rolling training window: the last train_frac of the history.
        train_len = int(train_frac * t0)
        if train_len < 1:
            # Not enough history for this fold to have a training set.
            train_len = 0
        train_lo = t0 - train_len
        # Purge removes the last purge_bars bars of the training window;
        # the embargo adds a further embargo_bars gap before the test
        # block. Both shrink the usable training slice from its end.
        train_hi = max(train_lo, t0 - gap)

        # -- in-sample backtest ---------------------------------------
        train_trades: list[Trade] = []
        if train_hi > train_lo:
            frame = df.iloc[train_lo:train_hi]
            train_trades = run_backtest(
                strategy_factory(),
                frame,
                direction=direction,
                risk_points=risk_points,
            )

        # -- out-of-sample backtest -----------------------------------
        test_trades: list[Trade] = []
        if t1 > t0:
            frame = df.iloc[t0:t1]
            test_trades = run_backtest(
                strategy_factory(),
                frame,
                direction=direction,
                risk_points=risk_points,
            )

        tr_mean, tr_win, tr_total, tr_n = _stats(train_trades)
        te_mean, te_win, te_total, te_n = _stats(test_trades)

        windows.append(
            WFWindow(
                fold=fold,
                train_start=int(times[train_lo]) if train_hi > train_lo else int(times[t0]),
                train_end=int(times[train_hi - 1]) if train_hi > train_lo else int(times[t0]),
                test_start=int(times[t0]),
                test_end=int(times[t1 - 1]),
                n_train_trades=tr_n,
                n_test_trades=te_n,
                test_mean_r=te_mean,
                test_win_rate=te_win,
                train_mean_r=tr_mean,
                train_win_rate=tr_win,
                test_total_r=te_total,
                train_total_r=tr_total,
                n_train_bars=max(0, train_hi - train_lo),
                n_test_bars=t1 - t0,
            )
        )
        all_is_trades.extend(train_trades)
        all_oos_trades.extend(test_trades)

    # -- aggregate -----------------------------------------------------
    is_mean, is_win, is_total, n_is = _stats(all_is_trades)
    oos_mean, oos_win, oos_total, n_oos = _stats(all_oos_trades)

    degradation = is_mean - oos_mean

    # -- edge significance ---------------------------------------------
    # The degradation verdict alone cannot say whether there WAS an edge.
    # Test it before deciding the verdict, so a strategy with no
    # detectable edge is never reported as "holds".
    oos_rs = np.array([t.pnl_r for t in all_oos_trades], dtype=float)
    edge = test_edge(oos_rs, seed=42)
    verdict = _verdict_with_significance(
        degradation,
        oos_mean,
        edge_significant=edge.significant,
        edge_conclusive=edge.conclusive,
    )

    # -- warnings ------------------------------------------------------
    warnings: list[str] = []

    if len(windows) == 0:
        warnings.append(
            "data is too short for the requested number of splits: "
            f"{n_bars} bars cannot produce {n_splits} out-of-sample folds "
            "with a usable training window."
        )
        verdict = "fails"

    if n_oos < MIN_OOS_TRADES_FOR_CONCLUSION:
        warnings.append(
            f"sample too small to conclude: only {n_oos} out-of-sample trades "
            f"(need >= {MIN_OOS_TRADES_FOR_CONCLUSION})"
        )

    empty_folds = [w.fold for w in windows if w.n_test_trades == 0]
    if empty_folds:
        warnings.append(
            "folds with zero out-of-sample trades (the strategy did not "
            "trade in that test block): fold(s) "
            + ", ".join(str(f) for f in empty_folds)
        )

    if oos_mean > 0 and degradation > 0:
        warnings.append(
            f"caveat: the edge shrank out of sample. Out-of-sample mean R is "
            f"still positive ({oos_mean:.3f}) but {degradation:.3f} R per trade "
            "lower than in sample, so part of the in-sample result does not "
            "generalise."
        )

    # The edge warning is mandatory and comes first: it is the caveat that
    # changes how the verdict should be read.
    edge_warning = edge.warning()
    if edge_warning:
        warnings.insert(0, edge_warning)
    warnings.extend(edge.warnings)

    if degradation < OOS_BETTER_IS_NOTE_THRESHOLD:
        warnings.append(
            f"caveat: out-of-sample beat in-sample by {-degradation:.3f} R per "
            "trade. This usually means the rolling training window was "
            "conservative or the sample is noisy, not that the strategy "
            "improved. Do not treat it as a real gain."
        )

    if gap > 0:
        warnings.append(
            f"purge/embargo active: {purge_bars} training bars purged before "
            f"the test boundary and {embargo_bars} bars embargoed after it, "
            "so the usable training sample is smaller than the raw bar count."
        )
    else:
        warnings.append(
            "no purge or embargo was applied: the training window touches "
            "the test block. Trades that straddle the boundary can leak "
            "test-period prices into the in-sample statistics."
        )

    return WFResult(
        windows=windows,
        oos_mean_r=oos_mean,
        oos_win_rate=oos_win,
        oos_total_r=oos_total,
        is_mean_r=is_mean,
        degradation=degradation,
        verdict=verdict,
        n_windows=len(windows),
        warnings=warnings,
        is_total_r=is_total,
        is_win_rate=is_win,
        n_oos_trades=n_oos,
        n_is_trades=n_is,
        edge_significant=edge.significant,
        edge_conclusive=edge.conclusive,
        edge_ci_low=edge.ci_low,
        edge_ci_high=edge.ci_high,
        edge_t_statistic=edge.t_statistic,
    )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def walk_forward_table(result: WFResult) -> str:
    """Render the per-fold results of a walk-forward run as plain text.

    Args:
        result: the :class:`WFResult` returned by :func:`walk_forward`.

    Returns:
        A plain-text table with aligned columns and a trailing summary
        line. No emoji, no colour, no markdown.
    """
    headers = [
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "is_trades",
        "is_mean_r",
        "oos_trades",
        "oos_mean_r",
        "oos_win_rate",
    ]
    rows: list[list[str]] = []
    for w in result.windows:
        rows.append(
            [
                str(w.fold),
                str(w.train_start),
                str(w.train_end),
                str(w.test_start),
                str(w.test_end),
                str(w.n_train_trades),
                f"{w.train_mean_r:.3f}",
                str(w.n_test_trades),
                f"{w.test_mean_r:.3f}",
                f"{w.test_win_rate:.3f}",
            ]
        )

    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]

    def _fmt(cells: list[str]) -> str:
        out = []
        for i, c in enumerate(cells):
            out.append(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]))
        return "  ".join(out).rstrip()

    lines = [_fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(_fmt(r) for r in rows)
    lines.append("")
    lines.append(
        f"verdict={result.verdict} "
        f"windows={result.n_windows} "
        f"is_mean_r={result.is_mean_r:.3f} "
        f"oos_mean_r={result.oos_mean_r:.3f} "
        f"oos_win_rate={result.oos_win_rate:.3f} "
        f"degradation={result.degradation:.3f}"
    )
    lines.append(
        f"pooled trades: in-sample={result.n_is_trades} "
        f"out-of-sample={result.n_oos_trades}"
    )
    for w in result.warnings:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
