"""Request parser tests. FakeLLM is used throughout, so no API calls are made.
'Today' is fixed at Monday 28 September 2026 so the tests never depend on the real date."""

from datetime import date

from agents.llm import FakeLLM, LLMError
from agents.parser import ParsedRequest, run_parser

TODAY = date(2026, 9, 28)


def _parsed(**fields):
    """A ParsedRequest where everything not given is 'not mentioned'."""
    blank = dict(
        origin=None, destination=None, start_date=None, end_date=None, duration_days=None,
        travellers=None, budget_amount=None, currency=None, style=None, pace=None, interests=[],
    )
    blank.update(fields)
    return ParsedRequest(**blank)


def _run(raw, *answers):
    fake = FakeLLM(list(answers))
    return run_parser({"raw_request": raw}, fake, today=TODAY), fake


# ---------- Form path ----------
def test_form_request_skips_the_llm(lisbon_request):
    fake = FakeLLM([])
    out = run_parser({"request": lisbon_request}, fake, today=TODAY)

    assert fake.calls == []
    assert "request" not in out  # the existing request is left untouched
    assert out["trace"][0].status == "success"


# ---------- Happy path ----------
def test_full_text_request_is_parsed_and_completed_by_code():
    out, fake = _run(
        "4 days in Lisbon from 10 Oct, 2 of us, about £1,500, we love food and history",
        _parsed(destination="Lisbon", start_date="2026-10-10", duration_days=4, travellers=2,
                budget_amount=1500, currency="GBP", interests=["food", "history"]),
    )
    r = out["request"]
    assert (r.destination, r.start_date, r.end_date) == ("Lisbon", date(2026, 10, 10), date(2026, 10, 13))
    assert (r.travellers, r.budget_amount, r.currency) == (2, 1500, "GBP")
    assert r.assumptions == [
        "Assumed departing from London.",
        "Assumed a mid-range travel style.",
        "Assumed a balanced pace.",
    ]
    prompt = fake.calls[0]["user"]
    assert "Monday 28 September 2026" in prompt
    assert "<data>" in prompt


def test_explicit_end_date_wins_over_duration():
    out, _ = _run("Lisbon 10 to 12 Oct",
                  _parsed(destination="Lisbon", start_date="2026-10-10",
                          end_date="2026-10-12", duration_days=7))
    assert out["request"].end_date == date(2026, 10, 12)


def test_missing_optional_details_become_assumptions():
    out, _ = _run("Barcelona 10 to 12 Oct",
                  _parsed(destination="Barcelona", start_date="2026-10-10", end_date="2026-10-12"))
    assumptions = out["request"].assumptions
    assert "Assumed 1 traveller." in assumptions
    assert "No budget given, so costs are shown without a limit." in assumptions


# ---------- Questions instead of guesses ----------
def test_missing_destination_asks_where():
    out, _ = _run("somewhere sunny with my partner", _parsed())
    assert out["status"] == "failed"
    assert out["warnings"] == ["Where would you like to go?"]
    assert "request" not in out


def test_missing_dates_asks_when():
    out, _ = _run("Lisbon please", _parsed(destination="Lisbon"))
    assert out["warnings"] == ["Which dates would you like to travel?"]


def test_missing_length_asks_how_long():
    out, _ = _run("Lisbon from 10 Oct", _parsed(destination="Lisbon", start_date="2026-10-10"))
    assert "How many days is the trip" in out["warnings"][0]


def test_trip_longer_than_seven_days_asks_to_shorten():
    out, _ = _run("Lisbon for 8 days from 10 Oct",
                  _parsed(destination="Lisbon", start_date="2026-10-10", duration_days=8))
    assert out["warnings"] == ["Trips are limited to 7 days. Could you shorten the dates?"]


def test_past_dates_are_questioned():
    out, _ = _run("Lisbon 1 to 3 Sept",
                  _parsed(destination="Lisbon", start_date="2026-09-01", end_date="2026-09-03"))
    assert "in the past" in out["warnings"][0]


def test_non_gbp_budget_is_questioned():
    out, _ = _run("Lisbon 10 to 12 Oct, 2000 euros",
                  _parsed(destination="Lisbon", start_date="2026-10-10", end_date="2026-10-12",
                          budget_amount=2000, currency="EUR"))
    assert "GBP only" in out["warnings"][0]


# ---------- Failures ----------
def test_empty_text_asks_without_calling_the_llm():
    out, fake = _run("   ")
    assert fake.calls == []
    assert out["status"] == "failed"


def test_llm_failure_suggests_the_form():
    out, _ = _run("Lisbon next week", LLMError("timeout"))
    assert "try the form" in out["warnings"][0]