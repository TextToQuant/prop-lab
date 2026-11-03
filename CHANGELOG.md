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

## [0.1.0]

- Initial release: backtest engine, free market-data adapters, purged
  walk-forward validation, bootstrap Monte Carlo challenge simulation,
  pass-rate surfaces with per-axis provenance, and a labelled Markdown
  report.
