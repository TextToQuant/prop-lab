"""Attribution must be visible, not buried.

The pass-rate surface this project is built around came from DeltaTrend
Trading's public work. If the credit is only in a footnote, or only in a
file that is not shipped, then the project has taken an idea and hidden
its source. These tests pin the credit in the files a reader actually
reads.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_readme_credits_deltatrend_trading():
    """The README names DeltaTrend Trading and links the channel."""
    readme = _read("README.md")
    assert "DeltaTrend Trading" in readme
    assert "deltatrendtrading" in readme
    assert "youtube.com/@deltatrendtrading" in readme


def test_readme_has_a_prominent_inspiration_section():
    """The credit is in its own section, not only at the bottom."""
    readme = _read("README.md")
    assert "## Inspiration" in readme

    # The Inspiration section has to appear before the Attribution
    # section, so a reader meets the credit early.
    assert readme.index("## Inspiration") < readme.index("## Attribution")

    # And it has to say what the inspiration actually was, rather than
    # only naming it.
    inspiration = readme.split("## Inspiration", 1)[1].split("## ", 1)[0]
    assert "surface" in inspiration.lower()


def test_readme_attribution_section_names_deltatrend_trading():
    readme = _read("README.md")
    attribution = readme.split("## Attribution", 1)[1]
    assert "DeltaTrend Trading" in attribution


def test_prior_art_documents_deltatrend_as_the_origin():
    """The prior-art review names the origin of the idea."""
    prior_art = _read("docs/PRIOR-ART.md")
    assert "DeltaTrend Trading" in prior_art
    assert "deltatrendtrading" in prior_art
    # It must say the idea is his, not just that his videos exist.
    assert "origin of the idea" in prior_art.lower()


def test_report_provenance_names_deltatrend_trading():
    """The generated report carries the credit in its provenance section."""
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_report import report as report_fixture

    # The fixture is decorated with @pytest.fixture, so calling it
    # directly is how to get the built report outside pytest.
    text = report_fixture.__wrapped__()
    provenance = text.split("## 6. Provenance")[1]
    assert "DeltaTrend Trading" in provenance
    assert "deltatrendtrading" in provenance
