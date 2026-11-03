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


def _load_bars(path: str, *, timeframe: str | None = None):
    """Load bars from a CSV or Parquet file."""
    from prop_lab.data import assign_session_dates, load_bars

    bars = load_bars(path, timeframe=timeframe)
    if "session_date" not in bars.columns:
        bars["session_date"] = assign_session_dates(bars)
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

    wf = walk_forward(factory, bars, n_splits=args.splits, direction=args.direction)

    print(f"strategy : {strategy_cls.name}")
    print(f"data     : {args.data}  ({len(bars)} bars, {_data_range(bars)})")
    print()
    print(walk_forward_table(wf))
    print()
    print(f"in-sample mean R  : {wf.is_mean_r:+.3f}")
    print(f"out-of-sample mean R: {wf.oos_mean_r:+.3f}")
    print(f"degradation       : {wf.degradation:+.3f}")
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
    from prop_lab.surface import (
        risk_geometry_surface,
        rr_surface,
        surface_table,
        winrate_surface,
    )

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(args.data, timeframe=args.timeframe)
    strategy = strategy_cls(**_strategy_kwargs(strategy_cls, args))
    trades = run_backtest(strategy, bars, direction=args.direction)
    if not trades:
        print("no trades were produced, cannot build surfaces", file=sys.stderr)
        return 1

    pnl = np.array([t.pnl_r for t in trades])
    sessions = [t.session_date for t in trades]
    rules = get_preset(args.firm)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    surfaces = [
        risk_geometry_surface(
            pnl, rules, n_attempts=args.attempts, seed=args.seed
        ),
        rr_surface(
            pnl,
            rules,
            measured=False,
            n_attempts=args.attempts,
            seed=args.seed,
        ),
        winrate_surface(
            pnl, rules, n_attempts=args.attempts, seed=args.seed
        ),
    ]

    print(surface_table(surfaces))
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
    from prop_lab.report import build_report
    from prop_lab.surface import risk_geometry_surface, rr_surface, winrate_surface
    from prop_lab.validate import walk_forward

    strategy_cls = _load_strategy(args.strategy)
    bars = _load_bars(args.data, timeframe=args.timeframe)
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

    wf = walk_forward(factory, bars, n_splits=args.splits, direction=args.direction)
    mc = monte_carlo_challenge(
        pnl,
        rules,
        risk_geometry=args.geometry,
        n_attempts=args.attempts,
        seed=args.seed,
        session_dates=sessions,
        compare_iid=True,
    )
    surfaces = [
        risk_geometry_surface(pnl, rules, n_attempts=args.attempts, seed=args.seed),
        rr_surface(pnl, rules, measured=False, n_attempts=args.attempts, seed=args.seed),
        winrate_surface(pnl, rules, n_attempts=args.attempts, seed=args.seed),
    ]

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
    p_rp.add_argument("--out", default="reports")
    p_rp.set_defaults(func=cmd_report)

    p_st = sub.add_parser("strategies", help="list available strategies and presets")
    p_st.set_defaults(func=cmd_strategies)

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
