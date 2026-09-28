"""Activities Agent tests. FakeLLM is used throughout, so no API calls are made.

Most tests use a relaxed one-day trip, so the target pool size is
ceil(2 activities x 1 day x 1.5) = 3, which keeps the numbers easy to follow.
"""

from datetime import date

from agents.activities import ActivityDecision, ActivityPick, run_activities_agent
from agents.llm import FakeLLM, LLMError
from providers.mock import MockActivityProvider
from schemas.models import TripRequest

SATURDAY = date(2026, 10, 10)
MONDAY = date(2026, 10, 12)


def _request(day=SATURDAY, interests=("architecture",), destination="Barcelona"):
    return TripRequest(
        origin="London",
        destination=destination,
        start_date=day,
        end_date=day,
        pace="relaxed",
        interests=list(interests),
    )


def _decision(*ids, note="Good matches."):
    return ActivityDecision(
        picks=[ActivityPick(id=i, reason=f"LLM reason for {i}") for i in ids],
        note=note,
    )


def _ids(out):
    return [a.id for a in out["activity_candidates"]]


def _run(state, fake):
    return run_activities_agent(state, MockActivityProvider(), fake)


# ---------- Happy path ----------
def test_llm_picks_come_first_with_their_reasons():
    fake = FakeLLM([_decision("BCN-A01", "BCN-A02", "BCN-A11")])
    out = _run({"request": _request()}, fake)

    assert _ids(out)[:3] == ["BCN-A01", "BCN-A02", "BCN-A11"]
    first = out["activity_candidates"][0]
    assert first.reason == "LLM reason for BCN-A01"
    assert first.price_per_person == 26  # price from the provider, not the LLM
    assert out["trace"][0].status == "success"


def test_free_options_are_guaranteed_with_honest_reasons():
    fake = FakeLLM([_decision("BCN-A01", "BCN-A02", "BCN-A11")])  # all paid
    out = _run({"request": _request()}, fake)
    by_id = {a.id: a for a in out["activity_candidates"]}

    free = [a for a in out["activity_candidates"] if a.price_per_person == 0]
    assert len(free) >= 2
    # A04 has an 'architecture' tag, so it gets a factual interest-match reason
    assert by_id["BCN-A04"].reason == "Matches your interests: architecture."
    # A07 matches no interest, so it gets the code's note
    assert by_id["BCN-A07"].reason == "Added as a free option in case the budget is tight."


def test_every_candidate_has_a_reason():
    fake = FakeLLM([_decision("BCN-A01")])
    out = _run({"request": _request()}, fake)
    assert all(a.reason for a in out["activity_candidates"])


# ---------- LLM misbehaves ----------
def test_invented_ids_are_dropped():
    fake = FakeLLM([_decision("BCN-A01", "BCN-A99", "BCN-A02")])
    out = _run({"request": _request()}, fake)

    assert "BCN-A99" not in _ids(out)
    assert "Ignored 1 activity ID(s)" in out["trace"][0].warnings[0]


def test_short_list_is_topped_up_by_interest_score():
    fake = FakeLLM([_decision("BCN-A01")])  # only 1 pick, target is 3
    out = _run({"request": _request()}, fake)

    # Top-up order: most architecture matches first, then cheapest: A04 (free), A06 (9)
    assert _ids(out)[:3] == ["BCN-A01", "BCN-A04", "BCN-A06"]
    assert any("Topped up 2" in w for w in out["trace"][0].warnings)


def test_llm_failure_falls_back_to_interest_ranking():
    fake = FakeLLM([LLMError("timeout")])
    out = _run({"request": _request()}, fake)

    assert out["trace"][0].status == "partial"
    assert _ids(out)[:3] == ["BCN-A04", "BCN-A06", "BCN-A11"]
    assert all(a.reason for a in out["activity_candidates"])


# ---------- Filtering ----------
def test_activities_closed_every_trip_day_are_removed():
    # One-day trip on a Monday: Picasso Museum (A03) and MNAC (A10) are closed
    fake = FakeLLM([_decision("BCN-A01")])
    out = _run({"request": _request(day=MONDAY)}, fake)

    prompt = fake.calls[0]["user"]
    assert "BCN-A03" not in prompt and "BCN-A10" not in prompt
    assert "BCN-A03" not in _ids(out) and "BCN-A10" not in _ids(out)
    assert "BCN-A05" in prompt  # closed on Sundays only, so still offered on a Monday


def test_excluded_ids_are_never_offered():
    fake = FakeLLM([_decision("BCN-A02")])
    out = _run({"request": _request(), "excluded_ids": ["BCN-A01"]}, fake)

    assert "BCN-A01" not in fake.calls[0]["user"]
    assert "BCN-A01" not in _ids(out)


# ---------- Assumptions and failures ----------
def test_no_interests_is_recorded_as_an_assumption():
    fake = FakeLLM([_decision("BCN-A01", "BCN-A05", "BCN-A07")])
    out = _run({"request": _request(interests=())}, fake)

    assert "balanced first-visit mix" in out["trace"][0].assumptions[0]
    assert "none given" in fake.calls[0]["user"]


def test_unsupported_destination_fails_without_calling_the_llm():
    fake = FakeLLM([])
    out = _run({"request": _request(destination="Paris")}, fake)

    assert out["activity_candidates"] == []
    assert out["trace"][0].status == "failed"
    assert "Demo data covers" in out["warnings"][0]
    assert fake.calls == []