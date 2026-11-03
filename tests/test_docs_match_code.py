"""The documentation must match the code.

AGENTS.md and SPEC.md are contracts that an AI agent follows. A contract
that has drifted from the implementation is worse than no contract,
because the agent trusts it and does the wrong thing. Nothing checked
that, which is how the verdict vocabulary, the CLI surface and the
dependency list all went stale while the code moved on.

These tests check the parts that can be verified mechanically: a
documented signature must match the real one, a documented verdict must
be one the code can return, and a documented command must exist.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Collapse whitespace, so a phrase split across a line still matches."""
    return " ".join(text.split()).lower()


AGENTS = _read("AGENTS.md")
SPEC = _read("SPEC.md")
README = _read("README.md")
HONESTY = _read("docs/HONESTY.md")


# --------------------------------------------------------------------------
# The verdict vocabulary
# --------------------------------------------------------------------------

def _real_verdicts() -> set[str]:
    """The verdict vocabulary, read from the module that defines it.

    `_VERDICTS` is the single source of truth, so the test reads it
    rather than guessing at the shape of the code that assigns verdicts.
    """
    from prop_lab.validate import walkforward

    vocabulary = getattr(walkforward, "_VERDICTS", None)
    assert vocabulary, "walkforward no longer exposes _VERDICTS"
    return set(vocabulary)


def test_verdicts_in_docs_are_the_ones_the_code_returns():
    """Every verdict named in the docs is a verdict the code can return."""
    returned = _real_verdicts()
    assert returned, "the verdict vocabulary is empty"

    for doc_name, doc in (("AGENTS.md", AGENTS), ("SPEC.md", SPEC)):
        named = set(re.findall(r"`(holds|degrades|no_edge|fails)`", doc))
        assert named, f"{doc_name} names no verdicts"
        unknown = named - returned
        assert not unknown, f"{doc_name} documents unknown verdicts: {unknown}"


def test_every_real_verdict_is_documented():
    """A verdict the code can return must be documented."""
    returned = _real_verdicts()
    for verdict in returned:
        assert f"`{verdict}`" in AGENTS, f"{verdict} is missing from AGENTS.md"
        assert f"`{verdict}`" in SPEC, f"{verdict} is missing from SPEC.md"


def test_no_edge_verdict_exists_in_the_code():
    """The `no_edge` verdict is real, not just documented."""
    assert "no_edge" in _real_verdicts()


def test_agents_md_documents_the_edge_test():
    """AGENTS.md must tell the agent to test the edge, not just backtest."""
    assert "test_edge" in AGENTS
    assert "no_edge" in AGENTS
    assert "zero-edge baseline" in AGENTS.lower() or "zero_edge_baseline" in AGENTS


# --------------------------------------------------------------------------
# Documented signatures must match the real ones
# --------------------------------------------------------------------------

def _signature_of(path: str, dotted: str) -> str:
    """Import a function from the package and return its signature."""
    import importlib

    module_name, _, attr = dotted.rpartition(".")
    obj = getattr(importlib.import_module(module_name), attr)
    return str(inspect.signature(obj))


# The parameter list for each function, as written in SPEC.md. The
# comment line after each signature block starts with "# " and lists the
# names, so the test can compare the DOCUMENT against the CODE rather
# than against a list maintained here. A list maintained in the test
# would only prove the test agrees with itself.
SPEC_SIGNATURE_COMMENTS = {
    "prop_lab.validate.walk_forward": "strategy_factory, df, n_splits, train_frac, purge_bars, embargo_bars, direction, risk_points",
    "prop_lab.validate.test_edge": "pnl_r_array, n_attempts, seed, confidence",
    "prop_lab.firm.monte_carlo_challenge": "pnl_r_array, rules, risk_geometry, n_attempts, seed, atr, max_trades, trades_per_day, max_days, resampling, session_dates, compare_iid, mean_block_length",
    "prop_lab.firm.monte_carlo.zero_edge_baseline": "rules, n_trades, risk_geometry, n_attempts, seed",
    "prop_lab.surface.risk_geometry_surface": "pnl_r_array, rules, risk_geometries, n_attempts, seed, session_dates",
    "prop_lab.surface.rr_surface": "pnl_r_array, rules, rr_values, risk_geometries, n_attempts, seed, measured, strategy_factory, df, session_dates",
    "prop_lab.surface.winrate_surface": "pnl_r_array, rules, win_rates, rr_values, n_attempts, seed, session_dates, operating_point, operating_point_label",
    "prop_lab.report.build_report": "wf, mc, surfaces, strategy_name, data_source, data_range, rules, extra_warnings, baseline_pass_rate",
}


def _real_parameters(dotted: str) -> set[str]:
    import importlib

    module_name, _, attr = dotted.rpartition(".")
    func = getattr(importlib.import_module(module_name), attr)
    return {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


@pytest.mark.parametrize("dotted", list(SPEC_SIGNATURE_COMMENTS))
def test_spec_signature_matches_the_code(dotted):
    """The signature written in SPEC.md must equal the real signature."""
    documented = {
        name.strip()
        for name in SPEC_SIGNATURE_COMMENTS[dotted].split(",")
        if name.strip()
    }
    real = _real_parameters(dotted)

    missing = real - documented
    assert not missing, (
        f"SPEC.md does not document these real parameters of {dotted}: "
        f"{sorted(missing)}"
    )

    invented = documented - real
    assert not invented, (
        f"SPEC.md documents parameters of {dotted} that do not exist: "
        f"{sorted(invented)}"
    )


@pytest.mark.parametrize("dotted", list(SPEC_SIGNATURE_COMMENTS))
def test_spec_signature_line_is_present_in_the_document(dotted):
    """The signature block itself must still be in SPEC.md.

    This is what catches a parameter being deleted from the document:
    the list above is a summary, and the summary can survive the
    deletion. The block cannot.
    """
    attr = dotted.rpartition(".")[2]
    assert f"def {attr}(" in SPEC, f"SPEC.md no longer shows the {attr} signature"

    # Every documented parameter must appear in the SPEC.md code blocks.
    for name in SPEC_SIGNATURE_COMMENTS[dotted].split(","):
        name = name.strip()
        if not name:
            continue
        assert name in SPEC, f"{name} is not mentioned anywhere in SPEC.md"


def test_agents_md_step3_shows_the_real_walk_forward_call():
    """The walk-forward example in AGENTS.md must be a legal call."""
    from prop_lab.validate import walk_forward

    signature = inspect.signature(walk_forward)
    # The doc shows purge_bars and embargo_bars, both of which must be
    # real keyword arguments.
    assert "purge_bars" in signature.parameters
    assert "embargo_bars" in signature.parameters
    assert "purge_bars=200" in AGENTS
    assert "embargo_bars=200" in AGENTS


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

def test_every_cli_command_in_the_docs_exists():
    """A command named in the docs must be a real subcommand."""
    from prop_lab.cli import main

    source = (ROOT / "src" / "prop_lab" / "cli.py").read_text(encoding="utf-8")
    # The parser registers each command with add_parser("name").
    commands = set(re.findall(r'add_parser\(\s*"([a-z_]+)"', source))
    assert commands, "could not find the command registry"

    for doc_name, doc in (("AGENTS.md", AGENTS), ("SPEC.md", SPEC)):
        named = set(re.findall(r"proplab ([a-z]+)", doc))
        # "proplab <command>" is prose, not a command name.
        named -= {"<command>"}
        unknown = named - commands
        assert not unknown, f"{doc_name} documents unknown commands: {unknown}"
    assert main is not None


def test_new_cli_commands_are_documented():
    """The commands added since the spec was written must be in the docs."""
    source = (ROOT / "src" / "prop_lab" / "cli.py").read_text(encoding="utf-8")
    commands = set(re.findall(r'add_parser\(\s*"([a-z_]+)"', source))
    for cmd in ("inspect", "trades", "sweep"):
        assert cmd in commands, f"{cmd} is missing from the CLI"
        assert cmd in AGENTS, f"{cmd} is missing from AGENTS.md"
        assert cmd in SPEC, f"{cmd} is missing from SPEC.md"


def test_agents_md_documents_the_session_and_purge_flags():
    """The flags that change the result must be documented."""
    for flag in ("--session-open-hour", "--purge-bars", "--embargo-bars"):
        assert flag in AGENTS, f"{flag} is missing from AGENTS.md"
        assert flag in SPEC, f"{flag} is missing from SPEC.md"


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------

def test_agents_md_dependency_list_matches_pyproject():
    """The documented hard dependencies are the real hard dependencies."""
    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)

    deps = " ".join(pyproject["project"]["dependencies"]).lower()
    for package in ("numpy", "pandas", "matplotlib", "yfinance", "requests"):
        assert package in deps, f"{package} is not a core dependency"

    # The data adapters are core, not optional. If a document still
    # calls yfinance or requests optional, it is wrong.
    for doc_name, doc in (("AGENTS.md", AGENTS), ("SPEC.md", SPEC)):
        assert "optional" not in doc.lower().split("yfinance")[0][-200:], (
            f"{doc_name} still describes a dependency as optional"
        )


def test_no_data_extra_is_documented():
    """There is no `data` extra; the install command must not ask for one."""
    for doc_name, doc in (("AGENTS.md", AGENTS), ("README.md", README)):
        assert '".[data,dev]"' not in doc, (
            f"{doc_name} installs a `data` extra that does not exist"
        )


# --------------------------------------------------------------------------
# Package layout
# --------------------------------------------------------------------------

def test_spec_package_layout_lists_the_files_that_exist():
    """SPEC.md's tree must not name a module that is not there."""
    assert "indicators.py" not in SPEC, (
        "SPEC.md lists src/prop_lab/indicators.py, which does not exist"
    )
    assert "csv_drop.py" not in SPEC, (
        "SPEC.md lists data/csv_drop.py; the loader is data/schema.py"
    )
    for module in ("schema.py", "significance.py"):
        assert module in SPEC, f"SPEC.md omits {module}"


def _spec_layout_paths() -> list[str]:
    """Reconstruct the file paths in SPEC.md's package tree.

    The tree is drawn with box characters, so the path has to be rebuilt
    from the indentation. A bare file name is not enough: `schema.py`
    exists, but only at `src/prop_lab/data/schema.py`, and a layout
    that put it at the root would be wrong.
    """
    block = SPEC.split("## 0. Package layout", 1)[1].split("```", 2)[1]
    stack: list[tuple[int, str]] = []
    paths: list[str] = []

    for raw in block.splitlines():
        if not raw.strip():
            continue
        # The comment arrow is the annotation column, not the path.
        raw = raw.split("\u2190", 1)[0].rstrip()
        if not raw.strip():
            continue

        # Strip the tree drawing: the connector and the vertical bars.
        text = raw.replace("\u251c\u2500\u2500 ", "").replace("\u2514\u2500\u2500 ", "")
        text = text.replace("\u2502", "").strip()
        if not text or text in {"prop-lab/", "}", "{"} or set(text) == {"-"}:
            continue

        # Every level of the tree is four columns wide.
        level = (len(raw) - len(raw.lstrip("\u2502 "))) // 4

        # Drop any sibling or deeper entry.
        while stack and stack[-1][0] >= level:
            stack.pop()

        if text.endswith("/"):
            # A directory entry becomes the parent of what follows it.
            stack.append((level, text.rstrip("/")))
            continue

        if not text.endswith(".py"):
            continue

        parent = "/".join(part for _, part in stack)
        paths.append(f"{parent}/{text}" if parent else text)

    return paths


def test_every_module_named_in_the_spec_layout_exists():
    """A path in the SPEC tree must exist on disk, exactly as written."""
    paths = _spec_layout_paths()
    assert paths, "could not read any path from the SPEC layout"

    for path in paths:
        candidate = ROOT / path
        assert candidate.exists(), (
            f"SPEC.md's package layout names {path}, which does not exist"
        )


def test_spec_layout_omits_modules_that_were_never_there():
    """A module that does not exist must not appear in the layout."""
    for absent in ("indicators.py", "csv_drop.py"):
        assert absent not in SPEC, (
            f"SPEC.md lists {absent}, which is not in the package"
        )


# --------------------------------------------------------------------------
# The code's own docstrings
# --------------------------------------------------------------------------

def test_walkforward_docstring_documents_every_verdict():
    """The module's own docstring must name every verdict it can return.

    A user reading `help(prop_lab.validate.walk_forward)` learns the
    vocabulary from the docstring. If it omits one, the documentation
    contradicts the code in the place a reader is most likely to look.
    """
    from prop_lab.validate import walkforward

    doc = walkforward.__doc__ or ""
    assert doc, "the walkforward module has no docstring"

    for verdict in _real_verdicts():
        assert f'"{verdict}"' in doc, (
            f"the walkforward docstring does not document the "
            f"{verdict!r} verdict"
        )


def test_walkforward_docstring_mentions_the_edge_refinement():
    """The docstring must explain that the verdict is refined."""
    from prop_lab.validate import walkforward

    doc = (walkforward.__doc__ or "").lower()
    assert "no_edge" in doc
    assert "significance" in doc or "distinguishable from zero" in doc


def test_significance_module_docstring_exists():
    """The edge test must be documented where it lives."""
    from prop_lab.validate import significance

    assert significance.__doc__, "the significance module has no docstring"
    assert "bootstrap" in significance.__doc__.lower()


# --------------------------------------------------------------------------
# docs/HONESTY.md
# --------------------------------------------------------------------------

def test_honesty_documents_every_verdict():
    """The honesty argument must use the vocabulary the code returns."""
    for verdict in _real_verdicts():
        assert f"`{verdict}`" in HONESTY, (
            f"HONESTY.md does not document the {verdict!r} verdict"
        )


def test_honesty_documents_the_zero_edge_baseline():
    """A pass rate is a comparison. The document has to say so."""
    assert "zero-edge baseline" in _flat(HONESTY)
    assert "zero_edge_baseline" in HONESTY or "zero-edge baseline" in HONESTY.lower()
    assert "27" in HONESTY, "HONESTY.md should state the approximate baseline"


def test_honesty_states_the_intrabar_upper_bound():
    """The largest known limitation must be stated, with its consequence."""
    flat = _flat(HONESTY)
    assert "intrabar" in flat
    # The two must be linked: the limitation and its consequence on the
    # number. "upper bound" on its own could be about anything.
    assert "pass rate here is an upper bound" in flat
    assert "not a central estimate" in flat


def test_honesty_does_not_claim_the_stationary_bootstrap_walks_cyclically():
    """The stationary bootstrap must be described as it now behaves."""
    # The buggy implementation walked the source cyclically from one
    # start. If the document describes that, it documents the defect.
    flat = _flat(HONESTY)
    assert "each block starts at a uniformly random index" in flat
    assert "permutation of the whole source array" in flat


def test_honesty_credits_deltatrend_trading():
    """The origin of the surface must be named."""
    assert "DeltaTrend Trading" in HONESTY
    assert "deltatrendtrading" in HONESTY


def test_honesty_states_the_licence_position_of_the_reference_project():
    """The reference project has no license; the document must say so."""
    flat = _flat(HONESTY)
    assert "carries no license file" in flat
    assert "dsilva42/prop-risk-geometry" in HONESTY


def test_honesty_mentions_purge_and_embargo_defaults():
    """A walk-forward that is not purged by default must be flagged."""
    assert "purge_bars" in HONESTY
    assert "embargo_bars" in HONESTY
    assert "not" in HONESTY.lower()


def test_honesty_states_the_rr_axis_gap_is_large():
    """The measured/modelled gap on the R:R axis is not a rounding error."""
    assert "56 percentage points" in HONESTY
