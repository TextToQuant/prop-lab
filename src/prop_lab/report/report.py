"""Build the labelled Markdown report.

The report is the deliverable a stranger's agent hands back to a human
who asked a single question: "what is my actual probability of passing
the evaluation, and does my edge survive out of sample?"

Its most important section is the second one, **Honesty**. That section
lists every axis of every surface with an explicit provenance tag --
``measured`` or ``modelled`` -- and it is never omitted, whatever the
data looks like. The rule is simple: an axis that was assumed must never
be presented as if it were measured.

The distinction the section explains:

- The **risk-geometry** axis can be measured, because the trade
  distribution is fixed by the backtest and only the position sizing
  changes from cell to cell. Every cell is a genuine simulation over the
  real trades.
- The **R:R** axis can be measured, but only by re-running the backtest
  at each R:R value, because changing the R:R changes which trades exist.
  When the cheap variant is used (winner rescaling), the axis is
  labelled ``modelled``.
- The **win-rate** axis cannot be measured from a backtest at all,
  because forcing a different win rate means fabricating trades that
  never happened. It is always ``modelled``.

The shape of the win-rate/R:R pass-rate surface was popularised by
DeltaTrend Trading on TikTok and YouTube using Topstep rules; the report
credits that line of work and states the caveat that its win-rate axis is
parametric by construction.

No emoji. Plain English. Numbers are printed, not described.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Iterable, Sequence

from ..surface.surfaces import AXIS_MEASURED, AXIS_MODELLED, Surface

__all__ = ["build_report", "CITATIONS", "STANDING_CAVEATS"]

#: The date on which every citation below was last checked.
CITATION_CHECKED_DATE = "2026-11-03"

#: Citations used by the provenance section of the report.
CITATIONS: list[tuple[str, str, str]] = [
    (
        "prop-risk-geometry (the reference project for risk geometries and "
        "the bootstrap Monte Carlo approach). The repository now lives at "
        "dsilva42/prop-risk-geometry; it was originally published as "
        "hunter547/prop-risk-geometry.",
        "https://github.com/dsilva42/prop-risk-geometry",
        CITATION_CHECKED_DATE,
    ),
    (
        "Bootstrap Monte Carlo resampling method (the statistical "
        "procedure used to resample the real per-trade distribution "
        "instead of flipping coins).",
        "https://doi.org/10.1080/01621459.1994.10476870",
        CITATION_CHECKED_DATE,
    ),
    (
        "Marcos Lopez de Prado, Advances in Financial Machine Learning, "
        "chapter 7 -- purged walk-forward cross-validation with embargo.",
        "https://github.com/hudson-and-thames/mlfinlab",
        CITATION_CHECKED_DATE,
    ),
    (
        "purgedcv, the scikit-learn-compatible purged and combinatorial "
        "cross-validation library by Evgenii Lazarev (eslazarev), whose "
        "purging and embargo conventions this package follows.",
        "https://github.com/eslazarev/purged-cross-validation",
        CITATION_CHECKED_DATE,
    ),
    (
        "DeltaTrend Trading -- the line of work that popularised the 3D "
        "pass-rate surface (pass rate as a function of win rate and R:R) "
        "for prop firm challenges, using Topstep rules. The win-rate axis "
        "of that surface is parametric by construction.",
        "https://www.youtube.com/@deltatrendtrading",
        CITATION_CHECKED_DATE,
    ),
]

#: Caveats that apply to every run, regardless of the data.
STANDING_CAVEATS: list[str] = [
    "Slippage and commission are not modelled. Fills are assumed to happen "
    "at the modelled price with no cost of doing the trade.",
    "News gaps are not modelled. A price gap that skips straight through a "
    "stop is not represented.",
    "Contract scaling plans and payout delays are not modelled.",
    "Firms change their rules, so the date the ruleset was checked "
    "matters; a pass rate computed against an old ruleset may not "
    "describe the firm today.",
    "The bootstrap approximates session boundaries with synthetic days, "
    "because a resampled trade sequence has no real calendar.",
    "A small trade sample cannot support a confident conclusion.",
]

_EMOJI_RE = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "]"
)


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def _fmt_float(value: Any, digits: int = 3) -> str:
    """Format a float for the report, tolerating ``None`` and non-numbers."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    """Format a proportion as a percentage string."""
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_money(value: Any) -> str:
    """Format a dollar amount with thousands separators."""
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_ts(value: Any) -> str:
    """Format a UNIX timestamp as 'YYYY-MM-DD HH:MM UTC'."""
    try:
        return dt.datetime.fromtimestamp(
            int(value), tz=dt.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return "n/a"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a plain Markdown table.

    Args:
        headers: the column headers.
        rows: the row values.

    Returns:
        A Markdown table string, or a short note when there are no rows.
    """
    body = [list(r) for r in rows]
    if not body:
        return "_No rows._\n"
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in body:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or mapping key from ``obj``, with a fallback."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ----------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------
def _headline(
    wf: Any, mc: Any, strategy_name: str, data_source: str, data_range: str, rules: Any
) -> str:
    """Build section 1, the headline.

    Args:
        wf: the walk-forward result.
        mc: the Monte Carlo summary.
        strategy_name: the strategy's display name.
        data_source: where the data came from.
        data_range: the date range covered by the data.
        rules: the firm ruleset.

    Returns:
        A Markdown section.
    """
    n_trades = _get(mc, "n_trades_sample", "n/a")
    pass_rate = _get(mc, "pass_rate", None)
    ci_low = _get(mc, "pass_rate_ci_low", None)
    ci_high = _get(mc, "pass_rate_ci_high", None)
    n_attempts = _get(mc, "n_attempts", "n/a")
    verdict = _get(wf, "verdict", "n/a")
    degradation = _get(wf, "degradation", None)
    rules_name = _get(rules, "name", type(rules).__name__ if rules else "n/a")

    lines = [
        "## 1. Headline",
        "",
        f"- Strategy: {strategy_name}",
        f"- Firm ruleset: {rules_name}",
        f"- Data source: {data_source}",
        f"- Data range: {data_range}",
        f"- Trades in the backtest sample: {n_trades}",
        f"- Monte Carlo attempts: {n_attempts}",
        f"- In-sample mean R: {_fmt_float(_get(wf, 'is_mean_r', None))}",
        f"- Out-of-sample mean R: {_fmt_float(_get(wf, 'oos_mean_r', None))}",
        f"- Out-of-sample win rate: {_fmt_pct(_get(wf, 'oos_win_rate', None))}",
        f"- Out-of-sample total R: {_fmt_float(_get(wf, 'oos_total_r', None))}",
        f"- Degradation (IS mean R minus OOS mean R): "
        f"{_fmt_float(degradation)}",
        f"- Pass rate: {_fmt_pct(pass_rate)} "
        f"(95% CI {_fmt_pct(ci_low)} to {_fmt_pct(ci_high)})",
        f"- Out-of-sample verdict: {verdict}",
        "",
    ]
    return "\n".join(lines)


def _honesty(surfaces: Sequence[Surface]) -> str:
    """Build section 2, the mandatory honesty table.

    Args:
        surfaces: the surfaces built for this run. May be empty, in which
            case the section says so explicitly rather than disappearing.

    Returns:
        A Markdown section.
    """
    lines = [
        "## 2. Honesty: what was measured and what was assumed",
        "",
        "Every axis of every surface carries a provenance tag. `measured` "
        "means the axis was computed from a real backtest or a real "
        "simulation over the real per-trade distribution. `modelled` means "
        "the axis was assumed or parametric, and is not the output of a "
        "backtest.",
        "",
    ]

    if not surfaces:
        lines.append("_No surfaces were built for this run, so there are no "
                     "axes to label._")
        lines.append("")
    else:
        for surface in surfaces:
            spec = surface.spec
            name = surface.axis_labels.get("name", spec.title)
            lines.append(f"### {name}")
            lines.append("")
            rows = []
            for axis_name, provenance, explanation in spec.provenance_rows():
                rows.append([axis_name, provenance, explanation])
            lines.append(_md_table(["axis", "provenance", "explanation"], rows))
            if spec.note:
                lines.append(spec.note)
                lines.append("")

    lines += [
        "### Why the distinction exists",
        "",
        "The risk-geometry axis can be measured because the trade "
        "distribution is fixed by the backtest and only the position "
        "sizing changes from cell to cell. Every cell of that surface is a "
        "genuine simulation over the real trades.",
        "",
        "The R:R axis can be measured, but only by re-running the "
        "backtest at each R:R value, because changing the R:R changes "
        "which trades exist: a trade that reached its target at 2R would "
        "not have reached it at 1R. When the R:R surface was produced by "
        "rescaling the winners of the existing trade array instead, the "
        "axis is labelled `modelled`, and the table above says so.",
        "",
        "The win-rate axis cannot be measured from a backtest at all, "
        "because forcing a different win rate means fabricating trades "
        "that never happened. Splitting a real trade distribution into "
        "wins and losses and re-drawing them at a different rate is a "
        "model, so the win-rate surface is labelled `modelled` on both "
        "of its axes, always.",
        "",
    ]
    return "\n".join(lines)


def _risk_geometry_table(surfaces: Sequence[Surface], mc: Any) -> str:
    """Build section 3, the per-risk-geometry pass-rate table.

    The rows are derived from any surface whose x axis is the risk
    geometry. When no such surface was supplied, the section says so
    plainly instead of inventing rows.

    Args:
        surfaces: the surfaces built for this run.
        mc: the Monte Carlo summary, used for the headline geometry when
            the caller did not supply a breakdown.

    Returns:
        A Markdown section.
    """
    lines = ["## 3. Risk geometry", ""]

    geom_surfaces = [
        s
        for s in surfaces
        if "risk geometry" in str(s.spec.x_name).lower()
        or "risk_geometry" in str(s.spec.x_name).lower()
    ]

    rows: list[list[Any]] = []
    for surface in geom_surfaces:
        spec = surface.spec
        for i, geometry in enumerate(spec.x_values):
            for j, ruleset in enumerate(spec.y_values):
                rate = float(surface.z[i, j])
                low = high = None
                if surface.ci_low is not None and surface.ci_high is not None:
                    low = float(surface.ci_low[i, j])
                    high = float(surface.ci_high[i, j])
                rows.append(
                    [
                        geometry,
                        ruleset,
                        _fmt_pct(rate),
                        _fmt_pct(low),
                        _fmt_pct(high),
                    ]
                )

    if rows:
        lines.append(
            _md_table(
                ["risk geometry", "ruleset", "pass rate", "CI low", "CI high"],
                rows,
            )
        )
        lines.append(
            "Confidence intervals are Wilson 95 percent intervals over the "
            "Monte Carlo attempts for that cell."
        )
        lines.append("")
    else:
        lines.append(
            "No per-risk-geometry breakdown was supplied for this run. The "
            "headline pass rate below comes from the single Monte Carlo "
            "simulation reported in section 1, at the default risk "
            "geometry."
        )
        lines.append("")
        pass_rate = _get(mc, "pass_rate", None)
        ci_low = _get(mc, "pass_rate_ci_low", None)
        ci_high = _get(mc, "pass_rate_ci_high", None)
        lines.append(
            f"- Pass rate: {_fmt_pct(pass_rate)} "
            f"(95% CI {_fmt_pct(ci_low)} to {_fmt_pct(ci_high)})"
        )
        lines.append("")
    return "\n".join(lines)


def _oos_table(wf: Any) -> str:
    """Build section 4, the per-fold walk-forward table.

    Args:
        wf: the walk-forward result.

    Returns:
        A Markdown section.
    """
    lines = ["## 4. Out-of-sample walk-forward", ""]
    windows = _get(wf, "windows", []) or []
    rows: list[list[Any]] = []
    for w in windows:
        rows.append(
            [
                _get(w, "fold", "n/a"),
                f"{_fmt_ts(_get(w, 'train_start', None))} to "
                f"{_fmt_ts(_get(w, 'train_end', None))}",
                f"{_fmt_ts(_get(w, 'test_start', None))} to "
                f"{_fmt_ts(_get(w, 'test_end', None))}",
                _get(w, "n_train_trades", "n/a"),
                _get(w, "n_test_trades", "n/a"),
                _fmt_float(_get(w, "test_mean_r", None)),
                _fmt_pct(_get(w, "test_win_rate", None)),
            ]
        )
    lines.append(
        _md_table(
            [
                "fold",
                "train bar range",
                "test bar range",
                "train trades",
                "test trades",
                "test mean R",
                "test win rate",
            ],
            rows,
        )
    )
    lines.append(f"- Folds: {_get(wf, 'n_windows', len(windows))}")
    lines.append(f"- Verdict: {_get(wf, 'verdict', 'n/a')}")
    lines.append("")
    return "\n".join(lines)


def _caveats(wf: Any, mc: Any, extra_warnings: Sequence[str] | None) -> str:
    """Build section 5, the caveats list.

    Args:
        wf: the walk-forward result.
        mc: the Monte Carlo summary.
        extra_warnings: caller-supplied extra warnings.

    Returns:
        A Markdown section.
    """
    lines = ["## 5. Caveats", ""]

    lines.append("### Warnings raised by this run")
    lines.append("")
    any_warning = False
    for label, source in (
        ("Walk-forward", _get(wf, "warnings", []) or []),
        ("Monte Carlo", _get(mc, "warnings", []) or []),
    ):
        for warning in source:
            any_warning = True
            lines.append(f"- {label}: {warning}")
    for warning in extra_warnings or []:
        any_warning = True
        lines.append(f"- Extra: {warning}")
    if not any_warning:
        lines.append("- No run-specific warnings were raised.")
    lines.append("")

    lines.append("### Standing caveats")
    lines.append("")
    for caveat in STANDING_CAVEATS:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _provenance() -> str:
    """Build section 6, the citations list."""
    lines = ["## 6. Provenance", ""]
    for text, url, checked in CITATIONS:
        lines.append(f"- {text} URL: {url}. Checked: {checked}.")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_report(
    *,
    wf: Any,
    mc: Any,
    surfaces: Sequence[Surface],
    strategy_name: str,
    data_source: str,
    data_range: str,
    rules: Any,
    extra_warnings: Sequence[str] | None = None,
) -> str:
    """Build the labelled Markdown report.

    The sections are always present, in this order, even when the data
    is thin: Headline, Honesty, Risk geometry, Out-of-sample, Caveats,
    Provenance. The Honesty section is mandatory and is never omitted.

    Args:
        wf: the walk-forward result. Needs ``is_mean_r``, ``oos_mean_r``,
            ``oos_win_rate``, ``oos_total_r``, ``degradation``,
            ``verdict``, ``n_windows``, ``warnings`` and ``windows``.
        mc: the Monte Carlo summary. Needs ``n_attempts``,
            ``pass_rate``, ``pass_rate_ci_low``, ``pass_rate_ci_high``,
            ``n_trades_sample`` and ``warnings``.
        surfaces: the surfaces built for this run. May be empty.
        strategy_name: the strategy's display name.
        data_source: where the data came from.
        data_range: the date range covered by the data.
        rules: the firm ruleset object. Needs at least a ``name``.
        extra_warnings: additional warnings to list under Caveats.

    Returns:
        The Markdown report as a string. No emoji.
    """
    sections = [
        "# Prop firm challenge report",
        "",
        f"Strategy: {strategy_name}",
        "",
        _headline(wf, mc, strategy_name, data_source, data_range, rules),
        _honesty(list(surfaces or [])),
        _risk_geometry_table(list(surfaces or []), mc),
        _oos_table(wf),
        _caveats(wf, mc, extra_warnings),
        _provenance(),
    ]
    report = "\n".join(sections).rstrip() + "\n"
    return report
