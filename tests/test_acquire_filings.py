"""The acquisition script's target cap: a cap selects the first N targets; 0 is rejected.

select_targets only parses arguments; nothing here calls SEC EDGAR.
"""

import pytest

from scripts.acquire_filings import TARGETS, select_targets


def test_no_cap_selects_every_target() -> None:
    assert select_targets([]) == TARGETS
    assert len(TARGETS) == 20


def test_cap_selects_the_first_targets() -> None:
    assert select_targets(["2"]) == TARGETS[:2]


@pytest.mark.parametrize("cap", ["0", "-3", "two"])
def test_invalid_cap_is_a_usage_error_not_a_full_run(cap: str) -> None:
    """A cap of 0 used to mean "no cap" and started the full 20-filing run."""
    with pytest.raises(SystemExit) as raised:
        select_targets([cap])
    assert raised.value.code == 2
