## What this changes

<!-- One or two sentences. What is different after this pull request? -->

## Why

<!-- The problem this solves, or the bug it fixes. -->

## How it was verified

<!-- Required. See CONTRIBUTING.md: a regression test must exercise the
     shipped code path, not a helper it happens to call. -->

- [ ] The offline test suite passes and still finishes in about ten seconds.
- [ ] A new or changed behaviour has a test that **fails without this
      change**. (Reintroduce the old behaviour, confirm the test fails,
      restore the change, confirm it passes.)
- [ ] I verified from a clean clone, not only from an editable install:
      ```bash
      git clone . /tmp/verify && cd /tmp/verify
      python -m venv --system-site-packages .venv
      .venv/bin/python -m pip install -e ".[dev]"
      .venv/bin/python -m pytest -q
      ```

## Honesty contract

- [ ] No warning was removed or filtered out.
- [ ] Any new surface axis is labelled `measured` or `modelled`, and the
      label is correct.
- [ ] No simulated trade outcome is drawn from anything but the real
      per-trade distribution.
- [ ] If a number's meaning changed, the report, the README and the
      changelog all say so.

## Public API

- [ ] No public API was renamed or removed. (`prop_lab.core`,
      `prop_lab.firm`, `prop_lab.surface`, `prop_lab.validate`,
      `prop_lab.report`, the `proplab` CLI, the
      `StrategyCore` / `Signal` / `Bar` contract, and the
      `measured` / `modelled` vocabulary are the product.)
- [ ] The fill model, the session rule semantics, the bootstrap method and
      the verdict vocabulary are unchanged.

## Attribution

- [ ] If this builds on another project, it is named, with its license.
