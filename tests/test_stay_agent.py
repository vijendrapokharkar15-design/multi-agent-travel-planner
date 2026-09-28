"""Stay Agent tests. FakeLLM is used throughout, so no API calls are made.

Reference facts for Barcelona (2 travellers = 1 room, 3 nights), with distance
measured to the top 6 activity candidates:
  BCN-S1 Gothic Quarter  155/night  ~0.8 km  late check-in yes
  BCN-S2 Eixample         68/night  ~1.1 km  late check-in NO
  BCN-S3 El Born         180/night  ~1.0 km  late check-in yes
  BCN-S4 Barceloneta     290/night  ~1.7 km  late check-in yes
"""

from datetime import date

from agents.llm import FakeLLM, LLMError
from agents.stay import StayDecision, StayReason, run_stay_agent
from providers.mock import MockActivityProvider, MockStayProvider
from schemas.models import TripRequest
from schemas.state import RevisionRequest

START = date(2026, 10, 10)
END = date(2026, 10, 13)


def _decision(best, selected, reasons=None):
    return StayDecision(
        best_value_id=best,
        selected_id=selected,
        reasons=[StayReason(id=i, reason=r) for i, r in (reasons or {}).items()],
    )


def _state(request, flight=None, activities=None, **extra):
    state = {
        "request": request,
        "activity_candidates": (
            activities if activities is not None
            else MockActivityProvider().search_activities("Barcelona")
        ),
    }
    if flight:
        state["flight_options"] = [flight]
        state["selected_flight_id"] = flight.id
    state.update(extra)
    return state


def _run(state, fake):
    return run_stay_agent(state, MockStayProvider(), fake)


def _labels(out):
    return {s.id: s.labels for s in out["stay_options"]}


# ---------- Same-day trip ----------
def test_same_day_trip_skips_everything():
    day_trip = TripRequest(origin="London", destination="Barcelona",
                           start_date=START, end_date=START)
    fake = FakeLLM([])
    out = _run(_state(day_trip), fake)

    assert out["stay_options"] == []
    assert out["selected_stay_id"] is None
    assert out["trace"][0].status == "success"
    assert fake.calls == []  # no LLM call needed


# ---------- Facts computed by code ----------
def test_cheapest_and_closest_are_labelled_by_code(barcelona_request):
    fake = FakeLLM([_decision("BCN-S1", "BCN-S1", {"BCN-S1": "Central and fairly priced."})])
    out = _run(_state(barcelona_request), fake)

    assert out["selected_stay_id"] == "BCN-S1"
    assert _labels(out)["BCN-S2"] == ["cheapest"]
    assert _labels(out)["BCN-S1"] == ["best_location", "best_value"]  # both labels kept

    prompt = fake.calls[0]["user"]
    assert "cheapest=BCN-S2" in prompt
    assert "closest to activities=BCN-S1" in prompt
    assert "GBP 465.00 total for 3 nights x 1 room(s)" in prompt  # 155 x 3, computed by code


# ---------- Hard filters ----------
def test_late_flight_removes_hotels_without_late_checkin(barcelona_request, bcn_late_flight):
    fake = FakeLLM([_decision("BCN-S1", "BCN-S1")])
    out = _run(_state(barcelona_request, flight=bcn_late_flight), fake)

    assert "BCN-S2" not in fake.calls[0]["user"]  # the LLM never sees it
    assert "BCN-S2" not in _labels(out)
    assert "Removed 1 option(s) without late check-in (flight lands 23:35)" in out["trace"][0].summary


def test_excluded_ids_are_never_offered(barcelona_request):
    fake = FakeLLM([_decision("BCN-S3", "BCN-S3")])
    out = _run(_state(barcelona_request, excluded_ids=["BCN-S1"]), fake)

    assert "BCN-S1" not in fake.calls[0]["user"]
    assert "best_location" in _labels(out)["BCN-S3"]  # next closest once S1 is excluded


def test_cheaper_stay_revision_only_offers_cheaper_options(barcelona_request):
    stays = MockStayProvider().search_stays("Barcelona", START, END)
    current = next(s for s in stays if s.id == "BCN-S3")  # 180 per night
    state = _state(
        barcelona_request,
        stay_options=[current],
        selected_stay_id="BCN-S3",
        revision_request=RevisionRequest(
            target="stay_agent",
            codes=["over_budget"],
            reason="Over by GBP 26",
            budget_action="cheaper_stay",
        ),
    )
    fake = FakeLLM([_decision("BCN-S1", "BCN-S1")])
    out = _run(state, fake)

    assert out["selected_stay_id"] == "BCN-S1"
    assert all(s.price_per_night < 180 for s in out["stay_options"])
    assert "Over by GBP 26" in fake.calls[0]["user"]


# ---------- LLM misbehaves ----------
def test_invented_id_falls_back_to_closest_affordable(barcelona_request):
    # Median nightly price is 167.50, so S1 (155) and S2 (68) are "affordable";
    # of those, S1 is closer to the activities.
    fake = FakeLLM([_decision("BCN-S99", "BCN-S99")])
    out = _run(_state(barcelona_request), fake)

    assert out["selected_stay_id"] == "BCN-S1"
    assert out["trace"][0].status == "partial"
    assert "unknown stay IDs" in out["trace"][0].warnings[0]


def test_llm_failure_falls_back_too(barcelona_request):
    fake = FakeLLM([LLMError("timeout")])
    out = _run(_state(barcelona_request), fake)

    assert out["selected_stay_id"] == "BCN-S1"
    assert out["trace"][0].status == "partial"


# ---------- Degrading gracefully ----------
def test_works_without_activity_locations(barcelona_request):
    fake = FakeLLM([LLMError("timeout")])
    out = _run(_state(barcelona_request, activities=[]), fake)

    # No distances, so the fallback picks the cheapest affordable option
    assert out["selected_stay_id"] == "BCN-S2"
    assert "distance was not considered" in out["trace"][0].summary


def test_unsupported_destination_fails_without_calling_the_llm():
    paris = TripRequest(origin="London", destination="Paris", start_date=START, end_date=END)
    fake = FakeLLM([])
    out = _run(_state(paris, activities=[]), fake)

    assert out["stay_options"] == []
    assert out["trace"][0].status == "failed"
    assert "Demo data covers" in out["warnings"][0]
    assert fake.calls == []