"""Figure guard tests: what counts as a stored figure, in chunks and in claims."""

from src.rag.figure_guard import PLACEHOLDER, withhold_figures

APPLE = frozenset({365_817_000_000.0, 94_680_000_000.0})
# BAC total assets stored 1,000x low; the loader adds the x1,000 variant.
BAC = frozenset({3_051_375_000.0, 3_051_375_000_000.0})


def test_table_figure_without_scale_word_is_masked() -> None:
    text, n = withhold_figures("Total net sales$365,817 33 %$274,515 6 %", APPLE)
    assert n == 1
    assert text == f"Total net sales${PLACEHOLDER} 33 %$274,515 6 %"


def test_answer_figure_with_scale_word_is_masked() -> None:
    text, n = withhold_figures("Net sales rose 33% to $365.8 billion, driven by iPhone.", APPLE)
    assert n == 1
    assert text == f"Net sales rose 33% to {PLACEHOLDER}, driven by iPhone."


def test_percentages_years_and_counts_pass() -> None:
    claim = "Services grew 27% in fiscal 2021; the Company had 154,000 employees."
    assert withhold_figures(claim, APPLE) == (claim, 0)


def test_unstored_amounts_pass() -> None:
    claim = "Prior-year net sales were $274.5 billion; buybacks were $85 billion."
    assert withhold_figures(claim, APPLE) == (claim, 0)


def test_scale_flagged_store_still_catches_the_true_figure() -> None:
    text, n = withhold_figures("Total assets reached $3.05 trillion.", BAC)
    assert n == 1
    assert PLACEHOLDER in text


def test_negative_in_parentheses_is_matched_on_magnitude() -> None:
    assert withhold_figures("Net income (94,680)", APPLE)[1] == 1


def test_no_targets_changes_nothing() -> None:
    assert withhold_figures("$365,817", frozenset()) == ("$365,817", 0)