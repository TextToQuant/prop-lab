# Contributing

Thanks for looking at this. It is research tooling and it is meant to be
improved in the open.

## What this project is

`prop-lab` answers one question for a retail trader:

> I have a trading idea. If I backtest it on real free market data and size
> it against a prop firm's rules, what is my actual probability of passing
> the evaluation, and does the edge survive out of sample?

The point is to answer that question *honestly*. A number that looks good
but came from a fabricated assumption is worse than no number, because the
person will act on it.

## The honesty contract

This is the part that matters, and it is enforced by the code rather than
by good intentions. If you touch any of the following, you are changing
the product's core value.

- **Bootstrap, never coin flip.** The outcome of every simulated trade is
  drawn from the real per-trade R-multiple array. If you see
  `rng.random() < win_rate` in a challenge simulation, the number is
  meaningless.
- **Block bootstrap by default.** Losses arrive in clusters and a trailing
  drawdown is a path-dependent barrier, so the order of trades matters. An
  i.i.d. bootstrap breaks the clusters apart and flatters the result. Pass
  `session_dates` and use `compare_iid=True` to see the gap.
- **Every surface axis is labelled.** The risk-geometry axis is measured:
  the trades are fixed and only the sizing changes, so every cell is a real
  simulation. The R:R axis is measured only if the backtest is re-run. The
  win-rate axis is **always modelled**, because forcing a different win
  rate means fabricating trades.
- **The sample size is printed.** Under 30 trades the pass rate is
  illustrative, not an estimate. Every pass rate comes with its 95 percent
  confidence interval.
- **The edge is tested for significance.** A pass rate is not a score. A
  strategy with no detectable edge still scores around a quarter on a
  typical 50K evaluation, because the target sits above the starting
  balance and the drawdown sits below it. The report states the zero-edge
  baseline and the verdict says `no_edge` when the confidence interval on
  the mean R includes zero.
- **Warnings are never filtered out.** Small samples, degenerate
  bootstraps, synthetic-day approximations, resampling gaps. They all go in
  the report.
- **The firm rules are dated.** Every preset records when it was checked
  against the firm's published terms. Re-verify them.

## Development setup

```bash
git clone https://github.com/<owner>/prop-lab.git
cd prop-lab
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Requires Python 3.10 or newer. Core dependencies are numpy, pandas and
matplotlib; `yfinance` and `requests` are needed only for the network data
sources.

## Running the tests

The **offline** suite must stay fast — around ten seconds. It is the
default suite and it must never touch the network. A live network test
once leaked into it and pushed the runtime from 9 seconds to 98; the CI
now has a step that fails the build if that happens again.

The network tests are gated behind an environment variable on purpose:

```bash
PROPLAB_NETWORK_TESTS=1 python -m pytest -q
```

One test needs the MNQ 5-minute dataset, which is not in the repository:

```bash
PROPLAB_MNQ_5M=/path/to/MNQ_5m.csv python -m pytest -q
```

## Rules for contributions

- **A regression test must exercise the shipped code path, not a helper it
  happens to call.** Prove each new test by reintroducing the bug and
  confirming it fails. This rule exists because an earlier version of the
  test suite passed against broken code.
- **Do not rename public API.** `prop_lab.core`, `prop_lab.firm`,
  `prop_lab.surface`, `prop_lab.validate`, `prop_lab.report`, the `proplab`
  CLI, the `StrategyCore` / `Signal` / `Bar` contract and the
  `measured` / `modelled` vocabulary are the product.
- **Do not change the fill model, the session rule semantics, the
  bootstrap method or the verdict vocabulary.** Those are the method. Only
  their application may be fixed.
- **Do not add dependencies** unless the alternative is genuinely worse.
- **Do not remove warnings.** Add them if they are missing.
- **Never trust an editable install.** Verify from a clean clone:

  ```bash
  git clone . /tmp/verify && cd /tmp/verify
  python -m venv --system-site-packages .venv
  .venv/bin/python -m pip install -e ".[dev]"
  .venv/bin/python -m pytest -q
  ```

  A bare `data/` pattern in `.gitignore` once excluded the real
  `src/prop_lab/data/` package from every clone while the editable install
  kept working locally. The clone test is how that was caught.

## Adding a strategy

Subclass `StrategyCore`, implement `on_bar`, return a `Signal`:

```python
from prop_lab.core import Bar, Signal, StrategyCore

class MyStrategy(StrategyCore):
    name = "my_strategy"

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        return Signal(
            direction="long",
            stop_distance=12.5,
            rr_ratio=2.0,
            reason="my_long_setup",
            bar_index=bar_index,
            timestamp=bar.timestamp,
        )
```

A signal on bar `i` is filled at the **open of bar `i+1`**. Every
assumption you made belongs in a `notes` string on the class, so the report
can print it.

## Adding a firm ruleset

Add it to `src/prop_lab/firm/rules.py` with a `source` URL and a
`rules_checked` date. Firms change their terms; a ruleset without a date
on it is a silent expiry waiting to happen.

## Attribution

If your work builds on another project, name it and name its license. See
`docs/PRIOR-ART.md` for what has already been credited and why.

## Not financial advice

This is research tooling. It produces a number under a stated set of
assumptions. It does not predict the future, and the assumptions are listed
in every report so you can judge them yourself.
