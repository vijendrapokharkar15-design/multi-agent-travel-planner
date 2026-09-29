"""Itinerary Agent tests. FakeLLM is used throughout, so no API calls are made.

Trip: London to Barcelona, Sat 10 to Tue 13 October 2026, hotel BCN-S1.
With flight BCN-F1 the usable windows are:
  Day 1 11:45-23:30 (lands 10:15 + 90 min), Days 2-3 09:30-23:30,
  Day 4 09:30-16:45 (3 hours before the 19:45 return).
"""

from datetime import date

import pytest

from agents.itinerary import (
    DayTheme,
    ItineraryDecision,
    fallback_assignments,
    run_itinerary_agent,
)
from agents.llm import FakeLLM, LLMError
from core.scheduler import Assignment, build_schedule
from core.validator import validate_plan
from providers.mock import MockFlightProvider, MockStayProvider
from schemas.models import WEEKDAYS
from schemas.state import BudgetAdvice, RevisionRequest

START = date(2026, 10, 10)
END = date(2026, 10, 13)


@pytest.fixture
def bcn_f1():
    flights = MockFlightProvider().search_flights("London", "Barcelona", START, END)
    return next(f for f in flights if f.id == "BCN-F1")


@pytest.fixture
def bcn_s1():
    stays = MockStayProvider().search_stays("Barcelona", START, END)
    return next(s for s in stays if s.id == "BCN-S1")


def _state(request, flight, stay, activities, **extra):
    state = {
        "request": request,
        "flight_options": [flight],
        "selected_flight_id": flight.id,
        "stay_options": [stay],
        "selected_stay_id": stay.id,
        "activity_candidates": activities,
    }
    state.update(extra)
    return state


def _a(day, slot, activity_id):
    return Assignment(day=day, slot=slot, activity_id=activity_id)


def _scheduled(out):
    return [i.activity_id for d in out["itinerary"] for i in d.items if i.kind == "activity"]


def _drop_paid_request(reason="Over by GBP 50"):
    return RevisionRequest(
        target="itinerary_agent", codes=["over_budget"],
        reason=reason, budget_action="drop_paid_activities",
    )


# ---------- Happy path ----------
def test_llm_plan_is_scheduled_with_themes(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    fake = FakeLLM([ItineraryDecision(
        assignments=[_a(2, "morning", "BCN-A01"), _a(2, "morning", "BCN-A02")],
        day_themes=[DayTheme(day=2, theme="Gaudi day")],
    )])
    out = run_itinerary_agent(_state(barcelona_request, bcn_f1, bcn_s1, bcn_activities), fake)

    assert _scheduled(out) == ["BCN-A01", "BCN-A02"]
    assert out["itinerary"][1].notes == "Gaudi day"
    assert out["itinerary"][0].notes is None
    assert out["trace"][0].status == "success"


def test_prompt_shows_each_days_usable_window(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    fake = FakeLLM([ItineraryDecision(assignments=[_a(2, "morning", "BCN-A01")], day_themes=[])])
    run_itinerary_agent(_state(barcelona_request, bcn_f1, bcn_s1, bcn_activities), fake)

    prompt = fake.calls[0]["user"]
    assert "Day 1 | Sat 10 Oct | usable 11:45-23:30" in prompt
    assert "Day 4 | Tue 13 Oct | usable 09:30-16:45" in prompt
    assert "Hotel area: Gothic Quarter" in prompt


def test_late_flight_days_are_marked_as_travel_days(
    barcelona_request, bcn_late_flight, bcn_s1, bcn_activities
):
    fake = FakeLLM([ItineraryDecision(assignments=[_a(2, "morning", "BCN-A01")], day_themes=[])])
    run_itinerary_agent(_state(barcelona_request, bcn_late_flight, bcn_s1, bcn_activities), fake)

    prompt = fake.calls[0]["user"]
    assert "Day 1 | Sat 10 Oct | no usable time (travel day)" in prompt
    assert "Day 4 | Tue 13 Oct | no usable time (travel day)" in prompt


# ---------- LLM misbehaves ----------
def test_useless_llm_plan_triggers_fallback(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    fake = FakeLLM([ItineraryDecision(
        assignments=[_a(2, "morning", "BCN-A98"), _a(3, "morning", "BCN-A99")],  # invented
        day_themes=[],
    )])
    out = run_itinerary_agent(_state(barcelona_request, bcn_f1, bcn_s1, bcn_activities), fake)

    assert out["trace"][0].status == "partial"
    assert any("scheduled no activities" in w for w in out["trace"][0].warnings)
    assert len(_scheduled(out)) > 0


def test_llm_failure_fallback_produces_a_valid_plan(
    barcelona_request, bcn_f1, bcn_s1, bcn_activities
):
    fake = FakeLLM([LLMError("timeout")])
    out = run_itinerary_agent(_state(barcelona_request, bcn_f1, bcn_s1, bcn_activities), fake)

    assert out["trace"][0].status == "partial"
    assert len(_scheduled(out)) > 0
    assert validate_plan(
        barcelona_request, bcn_f1, bcn_s1, out["itinerary"], bcn_activities, None
    ) == []


def test_fallback_respects_closed_days_and_pace(barcelona_request, bcn_f1, bcn_activities):
    assignments = fallback_assignments(barcelona_request, bcn_f1, bcn_activities)
    per_day: dict[int, int] = {}
    for a in assignments:
        per_day[a.day] = per_day.get(a.day, 0) + 1
        activity = next(x for x in bcn_activities if x.id == a.activity_id)
        day = START.toordinal() + a.day - 1
        assert WEEKDAYS[date.fromordinal(day).weekday()] not in activity.closed_days
    assert all(count <= 3 for count in per_day.values())  # balanced pace


# ---------- Revisions ----------
def test_drop_paid_revision_always_reduces_activity_cost(
    barcelona_request, bcn_f1, bcn_s1, bcn_activities
):
    plan = [_a(2, "morning", "BCN-A01"), _a(2, "morning", "BCN-A02"), _a(3, "morning", "BCN-A11")]
    previous = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, plan).days
    state = _state(
        barcelona_request, bcn_f1, bcn_s1, bcn_activities,
        itinerary=previous, revision_request=_drop_paid_request(),
    )
    # The LLM ignores the instruction and returns the same paid plan (26 + 35 + 13 = 74)
    fake = FakeLLM([ItineraryDecision(assignments=plan, day_themes=[])])
    out = run_itinerary_agent(state, fake)

    assert "BCN-A02" not in _scheduled(out)  # the priciest item on a busy day (35)
    assert "Removed Casa Batllo to reduce cost." in out["trace"][0].warnings
    prompt = fake.calls[0]["user"]
    assert "prefer free activities" in prompt
    assert "Over by GBP 50" in prompt


def test_drop_paid_meets_the_target_without_emptying_a_day(
    barcelona_request, bcn_f1, bcn_s1, bcn_activities
):
    # Sunday: Sagrada Familia (26) + Cathedral (9). Monday: Casa Batllo (35) on its own.
    plan = [_a(2, "morning", "BCN-A01"), _a(2, "morning", "BCN-A06"), _a(3, "morning", "BCN-A02")]
    previous = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, plan).days
    state = _state(
        barcelona_request, bcn_f1, bcn_s1, bcn_activities,
        itinerary=previous,
        revision_request=_drop_paid_request(),
        # 22.00 saving for 2 people incl. 10% contingency = 10.00 per person
        budget_advice=BudgetAdvice(action="drop_paid_activities", reason="x", estimated_saving=22.0),
    )
    fake = FakeLLM([ItineraryDecision(assignments=plan, day_themes=[])])  # LLM ignores the target
    out = run_itinerary_agent(state, fake)

    # Casa Batllo is the priciest overall, but removing it would empty Monday,
    # so code removes Sagrada Familia from busy Sunday instead
    assert _scheduled(out) == ["BCN-A06", "BCN-A02"]
    assert "Removed Sagrada Familia to reduce cost." in out["trace"][0].warnings
    assert "about GBP 10.00 per person" in fake.calls[0]["user"]


# ---------- Degrading gracefully ----------
def test_no_candidates_still_gives_travel_and_meals(barcelona_request, bcn_f1, bcn_s1):
    fake = FakeLLM([])
    out = run_itinerary_agent(_state(barcelona_request, bcn_f1, bcn_s1, []), fake)

    assert fake.calls == []  # nothing for the LLM to decide
    assert out["trace"][0].status == "partial"
    assert len(out["itinerary"]) == 4
    assert all(day.items for day in out["itinerary"])