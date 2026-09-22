"""Qualifier preservation — the WRONG/CORRECT pairs from the case brief, checked against evidence with the same phrasing."""

import pytest

from app.models.schemas import Evidence
from app.validation.qualifiers import check_qualifiers


def ev(eid: str, text: str) -> Evidence:
    return Evidence(id=eid, transcript_id=eid.split("_")[0], expert_name="X", role="R", market="M", speaker="X",
                    speaker_type="expert", timestamp="00:00", timestamp_seconds=0, text=text)


STRONGER_CENTRES = ev("a_05_07", "I would expect 15 to 20% more procedures each year in some of the stronger centres.")
SOME_AREAS = ev("c_04_06", "In some areas procedure growth could exceed 15% annually.")
WHOLE_MARKET = ev("b_05_08", "I would expect high single-digit or low double-digit growth in procedure volume, "
                             "rather than something like 20% across the whole market.")
FUNDING = ev("c_06_01", "If funding is already available it can be 6 to 9 months, but if we have to wait for the "
                        "capital cycle it can take much longer.")
FINANCE = ev("a_02_18", "Very important. The clinical argument may get surgeons interested, but the finance team "
                        "wants to understand utilisation, procedure volume and maintenance cost.")


@pytest.mark.parametrize("wrong, evidence, dropped", [
    ("France expects 15–20% annual growth.", STRONGER_CENTRES, "some of the stronger centres"),
    ("The UK expects >15% growth.", SOME_AREAS, "In some areas"),
    ("Germany expects 20% growth.", WHOLE_MARKET, "across the whole market"),
    ("UK purchasing takes 6–9 months.", FUNDING, "If funding is already available"),
    ("UK purchasing takes 6–9 months if funding is already available.", FUNDING, "can take much longer"),
])
def test_wrong_statements_are_flagged(wrong, evidence, dropped):
    warnings = check_qualifiers(wrong, [evidence])
    assert any(dropped.lower() in w.qualifier.lower() for w in warnings), warnings


@pytest.mark.parametrize("correct, evidence", [
    ("Dr. Martin expects 15–20% more procedures annually in some of the stronger centres.", STRONGER_CENTRES),
    ("Dr. Carter says procedure growth could exceed 15% annually in some areas.", SOME_AREAS),
    ("Anna Keller expects high single-digit or low double-digit procedure-volume growth rather than something "
     "like 20% across the whole market.", WHOLE_MARKET),
    ("Dr. Carter says 6–9 months if funding is already available, but it can take much longer when waiting for "
     "a capital cycle.", FUNDING),
])
def test_correct_statements_pass(correct, evidence):
    assert check_qualifiers(correct, [evidence]) == []


def test_universal_claim_is_flagged():
    warnings = check_qualifiers("Finance alone decides all purchases.", [FINANCE])
    assert {w.qualifier.lower() for w in warnings} >= {"alone", "all purchases"}


def test_balanced_emphasis_statement_passes():
    text = ("France and Germany place strong emphasis on the economic case, while the UK describes economics and "
            "clinical strategy as balanced.")
    assert check_qualifiers(text, [FINANCE]) == []


def test_forecast_horizon_numbers_do_not_trigger():
    assert check_qualifiers("Over the next 3–5 years she expects continued growth.", [SOME_AREAS]) == []
