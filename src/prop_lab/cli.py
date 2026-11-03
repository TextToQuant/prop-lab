"""Command line interface for prop-lab.

The one-shot path a stranger's agent runs::

    proplab report --strategy examples.strategies.trend_continuation \
        --data data/MNQ_5m.csv --firm apex-50k-eod --out reports/

Every subcommand prints plain text. No emoji, no colour codes.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd

__all__ = ["main", "build_parser"]

DEFAULT_STRATEGIES = {
    "trend_continuation": "examples.strategies.trend_continuation:TrendContinuation",
    "orb": "examples.strategies.opening_range_breakout:OpeningRangeBreakout",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_strategy(spec: str):
    """Load a strategy class from a ``module:Class`` string.

    The module may be a dotted path or a file path ending in ``.py``.

    Args:
        spec: the ``module:ClassName`` specification.

    Returns:
        The strategy class.

    Raises:
        ValueError: when the specification is malformed or the class is
            not found.
    """
    if ":" not in spec:
        raise ValueError(
            f"strategy must be given as 'module:ClassName', got {spec!r}"
        )
    module_path, class_name = spec.split(":", 1)

    if module_path.endswith(".py"):
        path = Path(module_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise ValueError(f"strategy file not found: {path}")
        spec_obj = importlib.util.spec_from_file_location(path.stem, path)
        if spec_obj is None or spec_obj.loader is None:
            raise ValueError(f"could not load strategy file: {path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[path.stem] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    if not hasattr(module, class_name):
        raise ValueError(
            f"{module_path} has no attribute {class_name!r}"
        )
    return getattr(module, class_name)


def _load_bars(
    path: str,
    *,
    timeframe: str | None = None,
    session_open_hour: int | None = None,
):
    """Load bars from a CSV or Parquet file.

    Args:
        path: the file to load.
        timeframe: optional resample target, e.g. ``'1h'``.
        session_open_hour: the local hour at which the trading session
            rolls over. ``18`` is the futures convention (18:00 ET to
            17:00 ET); ``0`` is the correct value for a 24/7 market such
            as crypto, where the session is the UTC calendar day.

            ``None`` means "decide from the data": if the source column
            says ``binance`` the market trades around the clock and the
            session is the UTC day, otherwise the futures rule applies.
            Getting this wrong is not cosmetic -- it changes where the
            daily loss limit resets and where positions are force-closed.
    """
    from prop_lab.data import assign_session_dates, load_bars

    bars = load_bars(path, timeframe=timeframe)
    if "session_date" in bars.columns:
        return bars

    if session_open_hour is None:
        source = ""
        if "source" in bars.columns:
            try:
                source = str(bars["source"].iloc[0]).strip().lower()
            except (IndexError, KeyError):
                source = ""
        # A 24/7 market has no session open in the futures sense.
        session_open_hour = 0 if source == "binance" else 18

    bars["session_date"] = assign_session_dates(bars, session_open_hour=session_open_hour)
    return bars


def _data_range(bars) -> str:
    """Human-readable date range of a bar frame."""
    import datetime as dt

    lo = dt.datetime.fromtimestamp(int(bars["time"].min()), tz=dt.timezone.utc)
    hi = dt.datetime.fromtimestamp(int(bars["time"].max()), tz=dt.timezone.utc)
    return f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}"


def _print_trade_stats(stats: dict) -> None:
    """Print the trade statistics of a backtest."""
    print(f"trades        : {stats['n_trades']}")
    print(f"win rate      : {stats['win_rate'] * 100:.1f}%")
    print(f"mean R        : {stats['mean_r']:+.3f}")
    print(f"total R       : {stats['total_r']:+.3f}")
    print(f"profit factor : {stats['profit_factor']:.2f}")
    print(f"max DD (R)    : {stats['max_drawdown_r']:.2f}")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def _build_surfaces(
    *,
    strategy_cls,
    kwargs: dict,
    bars,
    pnl,
    sessions,
    rules,
    n_attempts: int,
    seed: int,
    strategy_rr_overridable: bool = True,
):
    """Build the three surfaces for a run, consistently.

    Every surface is built from the SAME trade array, with the SAME
    session labels, so the cells of one surface are comparable with the
    cells of another and with the headline Monte Carlo number. Before
    this existed, the surface layer dropped the session labels and the
    R:R surface was always the modelled approximation, so the report
    could show two different pass rates for the same strategy.

    The R:R surface is MEASURED when the strategy's reward-to-risk can be
    overridden, because that is the only way the axis means anything:
    changing the R:R changes which trades exist, so the trade sequence
    has to be regenerated. When the strategy cannot be re-run at a
    different R:R the modelled fallback is used, and its axis says so.

    Args:
        strategy_cls: the strategy class.
        kwargs: constructor kwargs for the strategy.
        bars: the bar frame.
        pnl: per-trade ``pnl_r`` from the backtest.
        sessions: per-trade session labels.
        rules: the firm ruleset.
        n_attempts: Monte Carlo attempts per cell.
        seed: RNG seed.
        strategy_rr_overridable: False when the caller already knows the
            strategy cannot take an ``rr_ratio``.

    Returns:
        A list of three :class:`~prop_lab.surface.surfaces.Surface`
        objects, plus a list of notes about what was and was not
        measured.
    """
    from prop_lab.surface import (
        risk_geometry_surface,
        rr_surface,
        winrate_surface,
    )

    notes: list[str] = []

    def factory():
        return strategy_cls(**kwargs)

    s1 = risk_geometry_surface(
        pnl, rules, n_attempts=n_attempts, seed=seed, session_dates=sessions
    )

    # -- R:R surface: measured when the strategy allows it ---------------
    s2 = None
    if strategy_rr_overridable:
        try:
            s2 = rr_surface(
                pnl,
                rules,
                measured=True,
                n_attempts=n_attempts,
                seed=seed,
                strategy_factory=factory,
                df=bars,
                session_dates=sessions,
            )
            notes.append(
                "R:R surface: measured. The backtest was re-run at every "
                "R:R value, so each cell simulates the trades that really "
                "existed at that R:R."
            )
        except ValueError as exc:
            notes.append(
                f"R:R surface: fell back to the modelled approximation "
                f"({exc}). The R:R axis is an assumption, not a backtest "
                f"result."
            )
    if s2 is None:
        s2 = rr_surface(
            pnl,
            rules,
            measured=False,
            n_attempts=n_attempts,
            seed=seed,
            session_dates=sessions,
        )
        notes.append(
            "R:R surface: modelled. The strategy's reward-to-risk cannot be "
            "overridden, so the winners of the existing trade array were "
            "rescaled instead of re-running the backtest. The R:R axis is "
            "an assumption, not a backtest result."
        )

    # -- win-rate surface: always modelled, but anchored ----------------
    win_rate = float((pnl > 0).mean()) if pnl.size else 0.0
    rr_ratio = _strategy_rr_ratio(strategy_cls, kwargs, pnl)
    s3 = winrate_surface(
        pnl,
        rules,
        n_attempts=n_attempts,
        seed=seed,
        session_dates=sessions,
        operating_point=(win_rate, rr_ratio) if rr_ratio else None,
        operating_point_label="this strategy",
    )
    notes.append(
        f"win-rate surface: modelled, always. It assumes a two-point trade "
        f"distribution and cannot be measured from a backtest. This "
        f"strategy's real win rate is {win_rate:.1%} and its real "
        f"reward-to-risk is {rr_ratio:.2f}; the surface marks that point "
        f"but the value there is NOT this strategy's measured pass rate."
    )

    return [s1, s2, s3], notes


def _strategy_rr_ratio(strategy_cls, kwargs: dict, pnl) -> float:
    """Best-effort reward-to-risk of the strategy that produced ``pnl``.

    Taken from the strategy's own configuration when it exposes one,
    because the R:R a strategy was configured with is not always the R:R
    its trades achieved (a target that was never reached does not count).
    """
    for key in ("rr_ratio",):
        if key in kwargs and kwargs[key]:
            try:
                return float(kwargs[key])
            except (TypeError, ValueError):
                pass
    try:
        instance = strategy_cls(**kwargs)
        value = getattr(instance, "rr_ratio", None)
        if value:
            return float(value)
    except Exception:
        pass
    # Fall back to the achieved ratio: mean win over mean loss.
    if pnl.size:
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if wins.size and losses.size:
            mean_win = float(wins.mean())
            mean_loss = abs(float(losses.mean()))
            if mean_loss > 0:
                return mean_win / mean_loss
    return 0.0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show a strategy's parameters, so a user can discover what to tune."""
    strategy_cls = _load_strategy(args.strategy)
    print(f"strategy : {strategy_cls.name}")
    print(f"class    : {strategy_cls.__name__}")
    print(f"module   : {args.strategy}")
    print()
    print(strategy_cls.params_help())
    notes = getattr(strategy_cls(**_strategy_kwargs(strategy_cls, args)), "notes", None)
    if notes:
        print()
        print("notes:")
        print(f"  {notes}")
    return 0


def cmd_trades(args: argparse.Namespace) -> int:
    """Export the trade list so the result can be audited."""
    from prop_lab.core import run_backtest

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(
        args.data,
        timeframe=args.timeframe,
        session_open_hour=getattr(args, "session_open_hour", None),
    )
    kwargs = _strategy_kwargs(strategy_cls, args)
    trades = run_backtest(strategy_cls(**kwargs), bars, direction=args.direction)
    if not trades:
        print("no trades were produced", file=sys.stderr)
        return 1

    rows = []
    for t in trades:
        rows.append(
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_distance": t.stop_distance,
                "rr_ratio": t.rr_ratio,
                "pnl_points": t.pnl_points,
                "pnl_r": t.pnl_r,
                "reason": t.reason,
                "session_date": t.session_date,
                "exit_reason": t.exit_reason,
            }
        )
    frame = pd.DataFrame(rows)

    if args.out:
        out_path = Path(args.out)
        if out_path.parent and str(out_path.parent) not in ("", "."):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        print(f"wrote {out_path}")
        print(f"trades   : {len(frame)}")
    else:
        print(frame.to_csv(index=False), end="")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Sweep strategy parameters and report the result of every combination.

    This is the "how do different parameters fit into the data" command.
    Every cell is a real backtest followed by a real Monte Carlo
    simulation over that cell's real trades, so the numbers are measured
    in the same sense as the risk-geometry surface.
    """
    import itertools

    import numpy as np

    from prop_lab.core import run_backtest, summarize_trades
    from prop_lab.firm import get_preset, monte_carlo_challenge
    from prop_lab.validate import test_edge

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(
        args.data,
        timeframe=args.timeframe,
        session_open_hour=getattr(args, "session_open_hour", None),
    )
    rules = get_preset(args.firm)

    # -- parse the parameter grid -------------------------------------
    import inspect

    signature = inspect.signature(strategy_cls.__init__)
    known = {
        name
        for name, p in signature.parameters.items()
        if name not in ("self", "kwargs") and p.kind != p.VAR_KEYWORD
    }

    grid: list[tuple[str, list[object]]] = []
    for spec in args.param or []:
        if "=" not in spec:
            print(
                f"error: --param {spec!r} is not in NAME=v1,v2 form",
                file=sys.stderr,
            )
            return 2
        name, _, raw = spec.partition("=")
        name = name.strip()
        if name not in known:
            print(
                f"error: {name!r} is not a parameter of "
                f"{strategy_cls.__name__}. Available: {', '.join(sorted(known))}",
                file=sys.stderr,
            )
            return 2
        values: list[object] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token) if _looks_int(token) else float(token))
            except ValueError:
                values.append(token)
        if not values:
            print(f"error: --param {spec!r} has no values", file=sys.stderr)
            return 2
        grid.append((name, values))

    if not grid:
        print(
            "error: nothing to sweep. Use --param NAME=v1,v2 (repeatable).",
            file=sys.stderr,
        )
        return 2

    base_kwargs = _strategy_kwargs(strategy_cls, args)
    combinations = list(itertools.product(*[values for _, values in grid]))

    print(f"strategy : {strategy_cls.name}")
    print(f"data     : {args.data}  ({len(bars)} bars, {_data_range(bars)})")
    print(f"firm     : {rules.name}")
    print(f"attempts : {args.attempts} per cell")
    print(f"cells    : {len(combinations)}")
    print()

    rows: list[dict] = []
    for combo in combinations:
        kwargs = dict(base_kwargs)
        for (name, _), value in zip(grid, combo):
            kwargs[name] = value
        try:
            trades = run_backtest(strategy_cls(**kwargs), bars, direction=args.direction)
        except (TypeError, ValueError) as exc:
            print(f"error: {name}={value} raised {exc}", file=sys.stderr)
            return 1
        if not trades:
            rows.append(
                {
                    **{name: value for name, value in zip([g[0] for g in grid], combo)},
                    "trades": 0,
                    "win_rate": 0.0,
                    "mean_r": 0.0,
                    "edge_ci_low": 0.0,
                    "edge_ci_high": 0.0,
                    "edge_significant": False,
                    "pass_rate": 0.0,
                }
            )
            continue
        stats = summarize_trades(trades)
        pnl = np.array([t.pnl_r for t in trades])
        sessions = [t.session_date for t in trades]
        mc = monte_carlo_challenge(
            pnl,
            rules,
            risk_geometry=args.geometry,
            n_attempts=args.attempts,
            seed=args.seed,
            session_dates=sessions,
        )
        edge = test_edge(pnl, seed=args.seed)
        rows.append(
            {
                **{name: value for name, value in zip([g[0] for g in grid], combo)},
                "trades": stats["n_trades"],
                "win_rate": stats["win_rate"],
                "mean_r": stats["mean_r"],
                "edge_ci_low": edge.ci_low,
                "edge_ci_high": edge.ci_high,
                "edge_significant": edge.significant,
                "pass_rate": mc.pass_rate,
            }
        )

    frame = pd.DataFrame(rows)
    header = " | ".join([g[0] for g in grid])
    print(f"{header} | trades | win% | mean R | edge 95% CI | pass rate")
    for row in rows:
        combo_text = " | ".join(str(row[g[0]]) for g in grid)
        ci = f"[{row['edge_ci_low']:+.3f}, {row['edge_ci_high']:+.3f}]"
        print(
            f"{combo_text} | {row['trades']:6d} | {row['win_rate']*100:4.1f} | "
            f"{row['mean_r']:+.4f} | {ci} | {row['pass_rate']*100:5.1f}%"
        )

    significant = sum(1 for r in rows if r["edge_significant"])
    print()
    print(
        f"{significant} of {len(rows)} cells have an edge that is "
        f"distinguishable from zero."
    )

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "sweep.csv"
        frame.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
    return 0


def _looks_int(token: str) -> bool:
    """True when the token is an integer literal."""
    try:
        int(token)
    except ValueError:
        return False
    return True


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download data from a free source into a local CSV."""
    from prop_lab.data import binance_src, dukascopy_src, yahoo_src

    if args.source == "binance":
        bars = binance_src.fetch(
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )
    elif args.source == "yahoo":
        bars = yahoo_src.fetch(
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )
    elif args.source == "dukascopy":
        bars = dukascopy_src.fetch(
            instrument=args.symbol,
            start=args.start,
            end=args.end,
            timeframe=args.interval,
        )
    else:
        print(f"unknown source {args.source!r}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.symbol.replace("=", "_").replace("/", "_")
    path = out_dir / f"{name}_{args.interval}.csv"
    bars.to_csv(path, index=False)

    print(f"wrote {path}")
    print(f"rows   : {len(bars)}")
    print(f"range  : {_data_range(bars)}")
    coverage = bars.attrs.get("coverage")
    if coverage:
        print(f"coverage: {coverage}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a backtest and print the trade statistics."""
    from prop_lab.core import run_backtest, summarize_trades

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(args.data, timeframe=args.timeframe)
    strategy = strategy_cls(**_strategy_kwargs(strategy_cls, args))

    trades = run_backtest(strategy, bars, direction=args.direction)
    stats = summarize_trades(trades)

    print(f"strategy : {strategy_cls.name}")
    print(f"data     : {args.data}  ({len(bars)} bars, {_data_range(bars)})")
    print(f"params   : {strategy_cls.params_help()}")
    print()
    _print_trade_stats(stats)
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    """Run purged walk-forward out-of-sample validation."""
    from prop_lab.core import run_backtest, summarize_trades
    from prop_lab.validate import walk_forward, walk_forward_table

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(args.data, timeframe=args.timeframe)

    def factory():
        return strategy_cls(**_strategy_kwargs(strategy_cls, args))

    wf = walk_forward(
        factory,
        bars,
        n_splits=args.splits,
        direction=args.direction,
        purge_bars=getattr(args, "purge_bars", 0),
        embargo_bars=getattr(args, "embargo_bars", 0),
    )

    print(f"strategy : {strategy_cls.name}")
    print(f"data     : {args.data}  ({len(bars)} bars, {_data_range(bars)})")
    print()
    print(walk_forward_table(wf))
    print()
    print(f"in-sample mean R  : {wf.is_mean_r:+.3f}")
    print(f"out-of-sample mean R: {wf.oos_mean_r:+.3f}")
    print(f"degradation       : {wf.degradation:+.3f}")
    print(f"edge 95% CI       : [{wf.edge_ci_low:+.4f}, {wf.edge_ci_high:+.4f}]")
    if wf.edge_conclusive and wf.edge_significant:
        print("edge significance : the out-of-sample edge IS distinguishable from zero")
    elif wf.edge_conclusive:
        print("edge significance : NO edge detected (the interval includes zero)")
    else:
        print("edge significance : INCONCLUSIVE (too few out-of-sample trades)")
    print(f"verdict           : {wf.verdict}")
    for warning in wf.warnings:
        print(f"warning: {warning}")
    return 0


def cmd_montecarlo(args: argparse.Namespace) -> int:
    """Run the bootstrap Monte Carlo challenge simulation."""
    import numpy as np

    from prop_lab.core import run_backtest
    from prop_lab.firm import get_preset, monte_carlo_challenge

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(args.data, timeframe=args.timeframe)
    strategy = strategy_cls(**_strategy_kwargs(strategy_cls, args))

    trades = run_backtest(strategy, bars, direction=args.direction)
    if not trades:
        print("no trades were produced, cannot simulate", file=sys.stderr)
        return 1

    pnl = np.array([t.pnl_r for t in trades])
    sessions = [t.session_date for t in trades]
    rules = get_preset(args.firm)

    mc = monte_carlo_challenge(
        pnl,
        rules,
        risk_geometry=args.geometry,
        n_attempts=args.attempts,
        seed=args.seed,
        session_dates=sessions,
        compare_iid=True,
    )

    print(f"strategy : {strategy_cls.name}")
    print(f"firm     : {rules.name}")
    print(f"trades   : {len(trades)}")
    print()
    print(mc.describe())
    print()

    from prop_lab.firm.monte_carlo import zero_edge_baseline

    baseline = zero_edge_baseline(
        rules,
        n_trades=min(2000, len(pnl)) if len(pnl) else 2000,
        risk_geometry=args.geometry,
        n_attempts=max(1000, args.attempts // 4),
        seed=args.seed,
    )
    print(
        f"zero-edge baseline: {baseline:.1%} (a strategy with no edge at all "
        f"scores this under these rules)"
    )
    print(
        f"your strategy      : {mc.pass_rate:.1%} "
        f"({mc.pass_rate - baseline:+.1%} versus the baseline)"
    )
    print()
    for warning in mc.warnings:
        print(f"warning: {warning}")
    return 0


def cmd_surface(args: argparse.Namespace) -> int:
    """Build the pass-rate surfaces and save them as PNG files."""
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from prop_lab.core import run_backtest
    from prop_lab.firm import get_preset
    from prop_lab.surface import surface_table

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(
        args.data,
        timeframe=args.timeframe,
        session_open_hour=getattr(args, "session_open_hour", None),
    )
    kwargs = _strategy_kwargs(strategy_cls, args)
    trades = run_backtest(strategy_cls(**kwargs), bars, direction=args.direction)
    if not trades:
        print("no trades were produced, cannot build surfaces", file=sys.stderr)
        return 1

    pnl = np.array([t.pnl_r for t in trades])
    sessions = [t.session_date for t in trades]
    rules = get_preset(args.firm)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    surfaces, notes = _build_surfaces(
        strategy_cls=strategy_cls,
        kwargs=kwargs,
        bars=bars,
        pnl=pnl,
        sessions=sessions,
        rules=rules,
        n_attempts=args.attempts,
        seed=args.seed,
    )

    print(surface_table(surfaces))
    print()
    for note in notes:
        print(f"note: {note}")
    print()

    for surface in surfaces:
        stem = surface.spec.x_name
        fig = surface.to_3d()
        path3d = out_dir / f"surface_{stem}_3d.png"
        fig.savefig(path3d, dpi=120, bbox_inches="tight")
        print(f"wrote {path3d}")

        fig2 = surface.to_heatmap()
        path2d = out_dir / f"surface_{stem}_heatmap.png"
        fig2.savefig(path2d, dpi=120, bbox_inches="tight")
        print(f"wrote {path2d}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Run the whole pipeline and write a labelled Markdown report."""
    import numpy as np

    from prop_lab.core import run_backtest
    from prop_lab.firm import get_preset, monte_carlo_challenge
    from prop_lab.firm.monte_carlo import zero_edge_baseline
    from prop_lab.report import build_report
    from prop_lab.validate import walk_forward

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(
        args.data,
        timeframe=args.timeframe,
        session_open_hour=getattr(args, "session_open_hour", None),
    )
    kwargs = _strategy_kwargs(strategy_cls, args)

    def factory():
        return strategy_cls(**kwargs)

    trades = run_backtest(strategy_cls(**kwargs), bars, direction=args.direction)
    if not trades:
        print("no trades were produced, cannot build a report", file=sys.stderr)
        return 1

    pnl = np.array([t.pnl_r for t in trades])
    sessions = [t.session_date for t in trades]
    rules = get_preset(args.firm)

    wf = walk_forward(
        factory,
        bars,
        n_splits=args.splits,
        direction=args.direction,
        purge_bars=getattr(args, "purge_bars", 0),
        embargo_bars=getattr(args, "embargo_bars", 0),
    )
    mc = monte_carlo_challenge(
        pnl,
        rules,
        risk_geometry=args.geometry,
        n_attempts=args.attempts,
        seed=args.seed,
        session_dates=sessions,
        compare_iid=True,
    )
    baseline = zero_edge_baseline(
        rules,
        n_trades=min(2000, len(pnl)) if len(pnl) else 2000,
        risk_geometry=args.geometry,
        n_attempts=max(1000, args.attempts // 4),
        seed=args.seed,
    )

    surfaces, surface_notes = _build_surfaces(
        strategy_cls=strategy_cls,
        kwargs=kwargs,
        bars=bars,
        pnl=pnl,
        sessions=sessions,
        rules=rules,
        n_attempts=args.attempts,
        seed=args.seed,
    )

    strategy_label = strategy_cls.name
    if kwargs:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        strategy_label = f"{strategy_label} ({rendered})"

    markdown = build_report(
        wf=wf,
        mc=mc,
        surfaces=surfaces,
        strategy_name=strategy_label,
        data_source=os.path.basename(args.data),
        data_range=_data_range(bars),
        rules=rules,
        extra_warnings=surface_notes,
        baseline_pass_rate=baseline,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text(markdown, encoding="utf-8")
    print(f"wrote {path}")
    print()
    print(markdown)
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    """List the strategies that ship with the repository."""
    from prop_lab.firm import list_presets

    print("example strategies:")
    for name, spec in DEFAULT_STRATEGIES.items():
        print(f"  {name:22} {spec}")
    print()
    print("firm presets:")
    for preset in list_presets():
        print(f"  {preset}")
    print()
    print("risk geometries:")
    from prop_lab.firm import RISK_GEOMETRIES

    for geometry in RISK_GEOMETRIES:
        print(f"  {geometry}")
    return 0


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def _strategy_kwargs(strategy_cls, args: argparse.Namespace) -> dict:
    """Build the strategy constructor kwargs from the CLI flags."""
    kwargs: dict = {}
    if getattr(args, "rr", None) is not None:
        kwargs["rr_ratio"] = args.rr
    if getattr(args, "window", None) is not None:
        kwargs["window_minutes"] = args.window
    if getattr(args, "direction", None) in ("long", "short"):
        kwargs["direction"] = args.direction
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``proplab`` command."""
    parser = argparse.ArgumentParser(
        prog="proplab",
        description="Prop firm challenge lab: backtest, validate out of sample, "
        "and estimate the probability of passing an evaluation.",
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(p, *, needs_data=True, needs_strategy=True):
        if needs_data:
            p.add_argument("--data", required=True, help="path to a CSV or Parquet bar file")
            p.add_argument("--timeframe", default=None, help="resample to this timeframe")
            p.add_argument("--direction", default="both", choices=("both", "long", "short"))
            p.add_argument(
                "--session-open-hour",
                type=int,
                default=None,
                help=(
                    "local hour at which the trading session rolls over. "
                    "18 is the futures convention (18:00 ET to 17:00 ET); "
                    "0 is correct for a 24/7 market such as crypto, where "
                    "the session is the UTC calendar day. Default: infer "
                    "from the data source, 0 for Binance and 18 otherwise."
                ),
            )
        if needs_strategy:
            p.add_argument(
                "--strategy",
                required=True,
                help="strategy as 'module:Class', or 'long'/'short' for a built-in",
            )
            p.add_argument("--rr", type=float, default=None, help="reward:risk override")
            p.add_argument("--window", type=int, default=None, help="window in minutes override")
        p.add_argument("--seed", type=int, default=42)
        return p

    p_fetch = sub.add_parser("fetch", help="download free market data")
    p_fetch.add_argument("--source", required=True, choices=("binance", "yahoo", "dukascopy"))
    p_fetch.add_argument("--symbol", required=True)
    p_fetch.add_argument("--interval", default="1h")
    p_fetch.add_argument("--start", default=None)
    p_fetch.add_argument("--end", default=None)
    p_fetch.add_argument("--out", default="data")
    p_fetch.set_defaults(func=cmd_fetch)

    p_bt = add_common(sub.add_parser("backtest", help="run a backtest"))
    p_bt.set_defaults(func=cmd_backtest)

    p_wf = add_common(sub.add_parser("walkforward", help="purged walk-forward validation"))
    p_wf.add_argument("--splits", type=int, default=5)
    p_wf.add_argument(
        "--purge-bars",
        type=int,
        default=0,
        help=(
            "bars to remove from the end of each training window before "
            "the test block, so trades that straddle the boundary cannot "
            "leak test-period prices into the in-sample statistics. "
            "Default 0 means no purge."
        ),
    )
    p_wf.add_argument(
        "--embargo-bars",
        type=int,
        default=0,
        help=(
            "additional bars between the end of the training window and "
            "the start of the test block. Default 0 means no embargo."
        ),
    )
    p_wf.set_defaults(func=cmd_walkforward)

    p_mc = add_common(sub.add_parser("montecarlo", help="bootstrap Monte Carlo"))
    p_mc.add_argument("--firm", default="apex-50k-eod")
    p_mc.add_argument("--geometry", default="frac_2pct")
    p_mc.add_argument("--attempts", type=int, default=10_000)
    p_mc.set_defaults(func=cmd_montecarlo)

    p_sf = add_common(sub.add_parser("surface", help="build pass-rate surfaces"))
    p_sf.add_argument("--firm", default="apex-50k-eod")
    p_sf.add_argument("--attempts", type=int, default=2_000)
    p_sf.add_argument("--out", default="reports")
    p_sf.set_defaults(func=cmd_surface)

    p_rp = add_common(sub.add_parser("report", help="run everything and write a report"))
    p_rp.add_argument("--firm", default="apex-50k-eod")
    p_rp.add_argument("--geometry", default="frac_2pct")
    p_rp.add_argument("--attempts", type=int, default=10_000)
    p_rp.add_argument("--splits", type=int, default=5)
    p_rp.add_argument(
        "--purge-bars",
        type=int,
        default=0,
        help="bars purged from the end of each training window. Default 0.",
    )
    p_rp.add_argument(
        "--embargo-bars",
        type=int,
        default=0,
        help="additional bars embargoed after the training window. Default 0.",
    )
    p_rp.add_argument("--out", default="reports")
    p_rp.set_defaults(func=cmd_report)

    p_st = sub.add_parser("strategies", help="list available strategies and presets")
    p_st.set_defaults(func=cmd_strategies)

    p_ins = sub.add_parser(
        "inspect",
        help="show a strategy's parameters and the assumptions it carries",
    )
    p_ins.add_argument(
        "--strategy",
        required=True,
        help="strategy as 'module:Class' or 'path/to/file.py:Class'",
    )
    p_ins.add_argument("--rr", type=float, default=None, help="reward:risk override")
    p_ins.add_argument("--window", type=int, default=None, help="window in minutes override")
    p_ins.set_defaults(func=cmd_inspect)

    p_tr = add_common(sub.add_parser("trades", help="export the trade list for auditing"))
    p_tr.add_argument(
        "--out",
        default=None,
        help="write the trades to this CSV path. Omit to print to stdout.",
    )
    p_tr.set_defaults(func=cmd_trades)

    p_sw = add_common(sub.add_parser("sweep", help="sweep strategy parameters"))
    p_sw.add_argument(
        "--param",
        action="append",
        default=None,
        metavar="NAME=v1,v2",
        help=(
            "a parameter to vary, e.g. --param body_ratio=0.05,0.1,0.2. "
            "Repeat the flag to vary more than one parameter. Every "
            "combination is backtested and simulated for real."
        ),
    )
    p_sw.add_argument("--firm", default="apex-50k-eod")
    p_sw.add_argument("--geometry", default="frac_2pct")
    p_sw.add_argument(
        "--attempts",
        type=int,
        default=2000,
        help=(
            "Monte Carlo attempts per cell. 2000 is the default because a "
            "sweep runs many cells; raise it for a final answer."
        ),
    )
    p_sw.add_argument("--out", default=None, help="directory to write sweep.csv into")
    p_sw.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``proplab`` console script.

    Args:
        argv: argument list, defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
