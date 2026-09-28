"""Flight Agent tests. FakeLLM is used throughout, so no API calls are made."""

from datetime import date

from agents.flight import FlightDecision, FlightReason, run_flight_agent
from agents.llm import FakeLLM, LLMError
from providers.mock import MockFlightProvider
from schemas.models import TripRequest
from schemas.state import RevisionRequest

START = date(2026, 10, 10)
END = date(2026, 10, 13)


def _decision(best, selected, reasons=None):
    return FlightDecision(
        best_value_id=best,
        selected_id=selected,
        reasons=[FlightReason(id=i, reason=r) for i, r in (reasons or {}).items()],
    )


def _labels(out):
    return {f.id: f.label for f in out["flight_options"]}


# ---------- Happy path ----------
def test_llm_choice_is_applied_to_real_data(barcelona_request):
    fake = FakeLLM([_decision("BCN-F1", "BCN-F1", {"BCN-F1": "Good times, fair price."})])
    out = run_flight_agent({"request": barcelona_request}, MockFlightProvider(), fake)

    assert out["selected_flight_id"] == "BCN-F1"
    assert out["flight_options"][0].id == "BCN-F1"  # selected comes first
    assert _labels(out) == {"BCN-F1": "best_value", "BCN-F3": "cheapest", "BCN-F2": "fastest"}

    f1 = out["flight_options"][0]
    assert f1.reason == "Good times, fair price."
    assert f1.price_per_person == 124  # price comes from the provider, not the LLM
    assert out["trace"][0].status == "success"


# ---------- LLM misbehaves ----------
def test_invented_id_triggers_fallback(barcelona_request):
    fake = FakeLLM([_decision("BCN-F99", "BCN-F99")])
    out = run_flight_agent({"request": barcelona_request}, MockFlightProvider(), fake)

    # Fallback rule: cheapest direct flight
    assert out["selected_flight_id"] == "BCN-F3"
    trace = out["trace"][0]
    assert trace.status == "partial"
    assert "unknown flight IDs" in trace.warnings[0]


def test_llm_failure_triggers_fallback(barcelona_request):
    fake = FakeLLM([LLMError("timeout")])
    out = run_flight_agent({"request": barcelona_request}, MockFlightProvider(), fake)

    assert out["selected_flight_id"] == "BCN-F3"
    assert out["trace"][0].status == "partial"


# ---------- Filtering ----------
def test_excluded_ids_are_never_offered(barcelona_request):
    fake = FakeLLM([_decision("BCN-F1", "BCN-F1")])
    state = {"request": barcelona_request, "excluded_ids": ["BCN-F3"]}
    out = run_flight_agent(state, MockFlightProvider(), fake)

    assert "BCN-F3" not in _labels(out)
    assert _labels(out)["BCN-F4"] == "cheapest"  # next cheapest after F3
    assert "BCN-F3" not in fake.calls[0]["user"]  # the LLM never even saw it


def test_cheaper_flight_revision_only_offers_cheaper_options(barcelona_request):
    flights = MockFlightProvider().search_flights("London", "Barcelona", START, END)
    current = next(f for f in flights if f.id == "BCN-F1")  # 124 per person
    state = {
        "request": barcelona_request,
        "flight_options": [current],
        "selected_flight_id": "BCN-F1",
        "revision_request": RevisionRequest(
            target="flight_agent",
            codes=["over_budget"],
            reason="Over by GBP 100",
            budget_action="cheaper_flight",
        ),
    }
    fake = FakeLLM([_decision("BCN-F4", "BCN-F4")])
    out = run_flight_agent(state, MockFlightProvider(), fake)

    assert out["selected_flight_id"] == "BCN-F4"
    assert all(f.price_per_person < 124 for f in out["flight_options"])
    assert "Over by GBP 100" in fake.calls[0]["user"]  # the LLM is told why


# ---------- Unsupported route ----------
def test_unsupported_destination_fails_cleanly():
    paris = TripRequest(origin="London", destination="Paris", start_date=START, end_date=END)
    fake = FakeLLM([])
    out = run_flight_agent({"request": paris}, MockFlightProvider(), fake)

    assert out["selected_flight_id"] is None
    assert out["trace"][0].status == "failed"
    assert "Demo data covers" in out["warnings"][0]
    assert fake.calls == []  # no wasted LLM call when there is nothing to choose from