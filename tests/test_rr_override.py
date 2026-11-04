"""The measured R:R surface must actually re-run at a different R:R.

This covers a defect found by reading a rendered plot: the R:R surface
was a perfectly flat plane while its title claimed the axis was
MEASURED. The cause was an ordering error in the override helper. It
wrote to ``strategy.params['rr_ratio']`` and returned immediately, but a
strategy reads ``self.rr_ratio``, which is a separate attribute set in
``__init__``. So every R:R produced the same strategy, the same trades,
and the same pass rate, and nothing in the picture said so.

The existing test used a stand-in with no ``params`` attribute, which
took the good branch. Every real ``StrategyCore`` subclass has both, so
the test double never resembled the object that was broken.
"""
from __future__ import annotations

import numpy as np

from prop_lab.core import Bar, Signal, StrategyCore
from prop_lab.surface.surfaces import _build_strategy_with_rr


class _ParamsStrategy(StrategyCore):
    """A strategy shaped like the real ones: params AND an attribute.

    This is the shape that was broken. ``StrategyCore.__init__`` records
    the kwargs into ``self.params``, and the subclass also stores the
    value it actually reads in ``on_bar``.
    """

    name = "params_strategy"

    def __init__(self, rr_ratio: float = 1.0, window: int = 5, **kwargs):
        super().__init__(rr_ratio=rr_ratio, window=window)
        self.rr_ratio = rr_ratio
        self.window = window

    def on_bar(self, bar: Bar, bar_index: int) -> Signal | None:
        return None


def test_override_updates_the_attribute_the_strategy_reads():
    """The effective R:R must change, not just the params mapping."""
    strategy = _build_strategy_with_rr(lambda: _ParamsStrategy(), 2.5)

    assert strategy.rr_ratio == 2.5, (
        "the attribute on_bar reads still holds the old R:R, so the "
        "backtest would run unchanged and the surface would be flat"
    )
    assert strategy.params["rr_ratio"] == 2.5, (
        "the params mapping and the attribute disagree"
    )


def test_override_preserves_other_constructor_arguments():
    """Rebuilding must not silently reset the user's configuration."""
    strategy = _build_strategy_with_rr(lambda: _ParamsStrategy(window=17), 3.0)

    assert strategy.window == 17
    assert strategy.rr_ratio == 3.0


def test_override_detects_an_attribute_that_cannot_be_set():
    """A readable R:R that refuses to change must raise, not pass silently.

    Shipping a flat axis labelled MEASURED is worse than failing, so the
    post-condition turns it into an error. The class here records no
    ``rr_ratio`` in ``params``, so the rebuild route cannot help, and its
    setter accepts the value and discards it.
    """

    class _Stubborn(StrategyCore):
        name = "stubborn"

        def __init__(self, **kwargs):
            super().__init__()
            self._rr = 1.0

        @property
        def rr_ratio(self) -> float:
            return self._rr

        @rr_ratio.setter
        def rr_ratio(self, value) -> None:
            # Accepts the assignment and ignores it.
            return None

    import pytest

    with pytest.raises(ValueError, match="did not take effect"):
        _build_strategy_with_rr(lambda: _Stubborn(), 3.0)


def test_two_different_rrs_produce_different_instances():
    """The whole point: asking for two R:R values gives two behaviours."""
    one = _build_strategy_with_rr(lambda: _ParamsStrategy(), 1.0)
    three = _build_strategy_with_rr(lambda: _ParamsStrategy(), 3.0)

    assert one.rr_ratio == 1.0
    assert three.rr_ratio == 3.0
    assert np.isclose(one.params["rr_ratio"], 1.0)
    assert np.isclose(three.params["rr_ratio"], 3.0)
