"""Budget Advisor tests. FakeLLM is used throughout, so no API calls are made.

Reference state (2 travellers, 3 nights, 1 room, budget GBP 1,200):
  selected flight BCN-F1 at 124 pp; cheaper option BCN-F3 at 79 pp
  selected hotel  BCN-S1 at 155/night; cheaper option BCN-S2 at 68/night
  scheduled paid activities: Sagrada Familia 26 + Casa Batllo 35 = 61 pp
Estimated savings (including 10% contingency):
  cheaper_flight        (124 - 79) x 2 x 1.1          =  99.00
  cheaper_stay          (155 - 68) x 3 x 1 x 1.1      = 287.10
  drop_paid_activities  61 x 2 x 1.1                  = 134.20
"""

from datetime import date

import pytest

from agents.budget_advisor import AdvisorDecision, possible_fixes, run_budget_advisor
from agents.llm import FakeLLM, LLMError
from core.budget import compute_budget
from core.scheduler import Assignment, build_schedule
from providers.mock import MockFlightProvider, MockStayProvider

START = date(2026, 10, 10)
END = date(2026, 10, 13)


@pytest.fixture
def over_budget_state(barcelona_request, bcn_activities):
    flights = MockFlightProvider().search_flights("London", "Barcelona", START, END)
    stays = MockStayProvider().search_stays("Barcelona", START, END)
    flight_options = [f for f in flights if f.id in ("BCN-F1", "BCN-F3")]
    stay_options = [s for s in stays if s.id in ("BCN-S1", "BCN-S2")]
    flight = next(f for f in flight_options if f.id == "BCN-F1")
    stay = next(s for s in stay_options if s.id == "BCN-S1")

    itinerary = build_schedule(barcelona_request, flight, stay, bcn_activities, [
        Assignment(day=2, slot="morning", activity_id="BCN-A01"),
        Assignment(day=2, slot="morning", activity_id="BCN-A02"),
    ]).days
    budget = compute_budget(barcelona_request, flight, stay, itinerary, bcn_activities)

    return {
        "request": barcelona_request,
        "flight_options": flight_options,
        "selected_flight_id": "BCN-F1",
        "stay_options": stay_options,
        "selected_stay_id": "BCN-S1",
        "activity_candidates": bcn_activities,
        "itinerary": itinerary,
        "budget": budget,
    }


def _savings(state):
    return {action: saving for action, (saving, _) in possible_fixes(state).items()}


# ---------- Facts computed by code ----------
def test_possible_fixes_and_savings(over_budget_state):
    assert over_budget_state["budget"].over_by > 0
    assert _savings(over_budget_state) == {
        "cheaper_flight": pytest.approx(99.00),
        "cheaper_stay": pytest.approx(287.10),
        "drop_paid_activities": pytest.approx(134.20),
    }


def test_budget_that_excludes_flights_never_offers_a_cheaper_flight(over_budget_state):
    request = over_budget_state["request"].model_copy(update={"budget_includes_flights": False})
    state = {**over_budget_state, "request": request}
    assert "cheaper_flight" not in _savings(state)


def test_already_cheapest_options_leave_only_activities(over_budget_state):
    state = {**over_budget_state, "selected_flight_id": "BCN-F3", "selected_stay_id": "BCN-S2"}
    assert list(_savings(state)) == ["drop_paid_activities"]


def test_nothing_to_cut_returns_no_action_without_calling_the_llm(over_budget_state):
    state = {
        **over_budget_state,
        "selected_flight_id": "BCN-F3",
        "selected_stay_id": "BCN-S2",
        "itinerary": [],
    }
    fake = FakeLLM([])
    out = run_budget_advisor(state, fake)

    assert out["budget_advice"].action is None
    assert fake.calls == []


# ---------- The LLM's choice ----------
def test_llm_choice_is_applied(over_budget_state):
    fake = FakeLLM([AdvisorDecision(action="drop_paid_activities", reason="Keep the central hotel.")])
    out = run_budget_advisor(over_budget_state, fake)

    advice = out["budget_advice"]
    assert advice.action == "drop_paid_activities"
    assert advice.reason == "Keep the central hotel."
    assert advice.estimated_saving == pytest.approx(134.20)
    assert out["trace"][0].status == "success"

    prompt = fake.calls[0]["user"]
    assert "Over budget by GBP" in prompt
    assert "cheaper_stay: saves up to GBP 287.10" in prompt


def test_impossible_choice_falls_back_to_largest_saving(over_budget_state):
    # BCN-F3 is already selected, so cheaper_flight is not possible here
    state = {**over_budget_state, "selected_flight_id": "BCN-F3"}
    fake = FakeLLM([AdvisorDecision(action="cheaper_flight", reason="Cheaper flight.")])
    out = run_budget_advisor(state, fake)

    assert out["budget_advice"].action == "cheaper_stay"  # 287.10 beats 134.20
    assert out["trace"][0].status == "partial"
    assert "not possible here" in out["trace"][0].warnings[0]


def test_llm_failure_falls_back_to_largest_saving(over_budget_state):
    fake = FakeLLM([LLMError("timeout")])
    out = run_budget_advisor(over_budget_state, fake)

    assert out["budget_advice"].action == "cheaper_stay"
    assert out["budget_advice"].reason.startswith("Largest available saving")
    assert out["trace"][0].status == "partial"