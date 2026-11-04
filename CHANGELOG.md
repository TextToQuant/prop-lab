# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Edge-significance test** (`prop_lab.validate.test_edge`). A bootstrap
  confidence interval on the mean R per trade, so a strategy whose edge is
  statistically indistinguishable from zero can be told apart from one
  that has an edge. The walk-forward verdict gains a fourth value,
  `no_edge`, for exactly that case.
- **Zero-edge baseline** (`prop_lab.firm.monte_carlo.zero_edge_baseline`).
  The pass rate a strategy with *no* edge scores under a given ruleset.
  Reported in the headline, because a pass rate is a comparison and the
  comparison point was missing. For the APEX 50K EOD rules that baseline
  is roughly 27 percent.
- **Edge significance in the report headline**, with an explicit statement
  that the pass-rate confidence interval is an interval on the Monte Carlo
  estimate and not an interval on having an edge.
- **New CLI commands**: `proplab inspect` (a strategy's parameters and the
  assumptions it carries), `proplab trades` (export the trade list so the
  result can be audited), `proplab sweep` (sweep strategy parameters; every
  cell is a real backtest and a real simulation).
- **New CLI flags**: `--session-open-hour`, `--purge-bars`,
  `--embargo-bars`.
- **Operating-point marker on surfaces.** A surface now records where the
  strategy actually sits on the grid and marks it, and says so explicitly
  when the strategy is off the grid entirely.
- CI, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE` and this changelog.
- **`prop_lab.surface.geometry`**: the mesh rule in one place. A surface needs
  at least two values on BOTH axes, because `plot_surface` builds no quads
  along an axis of length 1. `build_mesh_grid` widens a single-value axis into
  a ribbon without altering a number; `assert_mesh_is_drawable` rejects an
  empty one.
- **`assert_rendered_surface_has_geometry`**: every figure is checked for
  actual geometry before it is written. It draws the canvas first (mplot3d
  builds polygons at projection time) and inspects only the first axes,
  because the colour bar is itself built from collections and would otherwise
  let a blank plot pass.
- **`--firm` accepts a comma-separated list**, and `surface`/`report` default
  to all three presets, so the ruleset axis is a real axis out of the box.

### Fixed

- **The walk-forward verdict no longer says `holds` for a strategy with no
  edge.** The degradation-only verdict answered "did the edge survive out
  of sample" and could not answer "was there an edge". A strategy with an
  out-of-sample mean R of +0.009 and a confidence interval spanning zero
  was reported as `holds`, which reads as "the edge survived" when there
  was never an edge to survive.
- **The R:R surface is measured when the strategy allows it.** The CLI
  always rendered the modelled approximation, in which the winners of the
  existing trade array are rescaled and the trade sequence is not
  regenerated. Because changing the R:R changes which trades exist, the
  modelled surface reported pass rates up to 56 percentage points away
  from a real re-backtest at the same R:R. The CLI now re-runs the
  backtest at each R:R value and falls back to the modelled version only
  when the strategy cannot take an `rr_ratio`.
- **The surface layer and the headline Monte Carlo use the same
  resampling.** The surface layer dropped the per-trade session labels,
  so the block bootstrap degenerated toward i.i.d. and the report could
  show two different pass rates for the same strategy.
- **The CLI can express the correct session rule.** `assign_session_dates`
  defaults to the futures rule (18:00 ET to 17:00 ET). On a 24/7 market
  that is the wrong boundary, and the CLI had no way to say otherwise. The
  rule is now inferred from the data source (0 for Binance, 18 otherwise)
  and can be set explicitly with `--session-open-hour`.
- **The CLI can actually purge.** `walk_forward` implements purge and
  embargo and the engine warns when they are absent, but the command line
  never passed them, so the documented "purged walk-forward" was not
  purged. `--purge-bars` and `--embargo-bars` are now available on
  `walkforward` and `report`.
- **The 3D surface no longer renders empty.** `plot_surface` builds no quads
  when an axis has a single value and does not raise, so the figure came back
  with a correct title, colour bar and axis limits and an entirely blank
  plotting area. The risk-geometry surface has exactly that shape with one
  ruleset, which is the default, so the headline figure was empty.
- **The measured R:R surface was a flat plane labelled MEASURED.** The
  override helper wrote `strategy.params["rr_ratio"]` and returned, while
  every strategy reads `self.rr_ratio`. Two stores, one unused: the backtest
  ran unchanged at every R:R. It affected 2 of the 3 bundled examples; the
  third worked only because it never records `rr_ratio` in `params`. The
  helper now rebuilds through the constructor and asserts the effective
  value, raising if the override did not take.
- **3D axis labels collided with tick labels.** In mplot3d the axis label sits
  at the middle of the axis line, exactly where the tick labels are; pushing
  it out with `labelpad` walks it off the canvas instead. The labels are
  omitted on 3D and the title carries the axis names.
- **`tight_layout` no longer fails on the 3D figures.** It warned that the
  margins could not be accommodated and then did nothing; the save-time
  `bbox_inches="tight"` handles the margins.
- **The session window is measured in elapsed time, not in bars.** Both
  bundled session strategies converted `window_minutes` with a hard-coded
  `bars_per_minute = 5`, so "30 minutes" silently became **90 minutes** on
  15-minute data, and no CLI flag could correct it. The window now runs from
  the first bar of the session until `window_minutes` have actually elapsed,
  so it means the same thing on any timeframe. `bars_per_minute` is removed.
  This changes the bundled examples' published numbers: on the same BTCUSDT
  15m data the opening-range example moves from +0.002R to -0.174R mean R,
  which is the misconfiguration being removed rather than a regression.
- **The i.i.d. resampling diagnostic no longer inverts its own adjective.**
  `gap` is the block pass rate minus the i.i.d. one, so a negative gap means
  the i.i.d. run scored *higher* and is the optimistic reading. The code
  labelled that case "pessimistic", which told the reader the opposite of the
  truth about which resampling to trust.
- **`fails` and "no edge was detected" are reported as one statement.** They
  are two true statements about one result, but printed side by side they read
  as a contradiction: `fails` suggests an edge that broke, when there was
  never an edge. The warning and the report headline now join them explicitly.
- **The session rule is applied by every command, and it says which one it
  used.** `--session-open-hour` reached `surface` and `report` but was ignored
  by `backtest`, `walkforward` and `montecarlo`, so on a non-Binance CSV half a
  run used the futures day and the other half used the flag. All data commands
  now forward it, the loader infers it from the data when it is absent, and it
  prints `session rule: 0 (--session-open-hour)` or
  `session rule: 18 (inferred from source=unknown)` so a mismatch is visible.
- **The zero-edge baseline respects the strategy's bracket.** It was built from
  equal numbers of `+1R` and `-1R`, i.e. a hard-coded 1:1 with a 50 percent
  assumed hit rate, whatever the strategy traded. With no edge the chance of
  reaching the target before the stop is `1/(1+rr)` — 50.0 percent at 1:1,
  33.3 at 1:2, 16.7 at 1:5 — so a 2R strategy was being compared against a
  different game. `zero_edge_baseline` now takes `rr`, and the CLI passes the
  strategy's own bracket. A static-floor cross-check against the gambler's-ruin
  value of `2000/5000 = 40%` was added as a test.

### Changed

- The report's standing caveats now state that **the drawdown is checked
  on closed trade equity, not intrabar**, and that the pass rate is
  therefore an upper bound. The simulation sees one equity number per
  trade, so an account that dips below the trailing floor mid-trade and
  recovers into the close is scored as surviving. This is the largest
  known source of optimism in the simulator and it cannot be tuned away,
  because the bootstrap has no excursion data to work with.
- The provenance section credits the reference project for the *approach*
  and states explicitly that no code was taken from it, that the package is
  an independent implementation, and that the original repository URL no
  longer resolves.
- The provenance section adds the only public empirical pass-rate dataset
  found, so a user can see what real accounts score against what this
  simulator produces.
- **A 3D surface is written only when the grid varies on both axes**, and only
  for the two surfaces where 3D adds something: the ruleset x risk-geometry
  cross and win rate x R:R. The R:R x risk-geometry surface is a heatmap,
  because its cells are read one at a time. A grid with a constant axis is a
  list of numbers, not a surface, and is no longer shipped as a curtain.
- **`README.md` documents what the risk-geometry names mean.** `frac_2pct`,
  `dd_frac_40pct`, `anti_martingale`, `floor_aware` and `max_size` are each
  one sizing rule, and the axis was previously unreadable without the source.
- `AGENTS.md` records the 3D rule and the multi-firm `--firm` behaviour.
- **`README.md` states what the zero-edge baseline is and is not.** It is
  specific to the strategy's bracket (`1/(1+rr)`), so a 1.5R strategy is
  compared against a 1.5R baseline, not the 27% figure in the docs; and it
  assumes no time limit, because a bracket's hit rate depends on the time it is
  given while its expectation does not. The second point is a limitation, not an
  approximation, and it is now stated where the number is introduced.
- **`README.md` documents the `on_bar`-while-flat trap.** The engine calls
  `on_bar` only when no position is open, so an indicator folded bar by bar
  accumulates over a gapped stream and is wrong without failing. Two bundled
  strategies carry the latent problem. Documented rather than fixed: changing
  the engine would move every published number, including the reference
  baseline, and that is a separate decision.
- **`docs/images/` adds a figure gallery**, and the README shows it: the
  ruleset x risk-geometry cross (3D and heatmap), the win-rate x R:R surface
  (3D and heatmap) and the R:R x risk-geometry heatmap, all produced by one
  command on the bundled example.
- `README.md` no longer claims the core dependencies are only numpy, pandas and
  matplotlib; `yfinance` and `requests` are core, because the data adapters are.

## [0.1.0]

- Initial release: backtest engine, free market-data adapters, purged
  walk-forward validation, bootstrap Monte Carlo challenge simulation,
  pass-rate surfaces with per-axis provenance, and a labelled Markdown
  report.
