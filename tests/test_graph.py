"""End-to-end graph tests with FakeLLM (answers given per schema, so the
parallel Flight and Activities agents can run in any order). No API calls.

Trip: London to Barcelona, Sat 10 to Tue 13 October 2026, 2 travellers.
The itinerary uses only free activities, one on each full day (Sun and Mon),
so costs are easy to follow:
  with BCN-F1 + BCN-S1: (248 + 465 + 400 food + 128 transport) x 1.1 = 1,365.10
  with BCN-F1 + BCN-S2: (248 + 204 + 400 + 128) x 1.1                 = 1,078.00
"""

from datetime import date

import pytest

from agents.activities import ActivityDecision, ActivityPick
from agents.budget_advisor import AdvisorDecision
from agents.flight import FlightDecision
from agents.itinerary import ItineraryDecision
from agents.llm import FakeLLM
from agents.parser import ParsedRequest
from agents.stay import StayDecision
from core.scheduler import Assignment
from graph.build import plan_trip
from graph.nodes import router_node
from schemas.models import TripRequest
from schemas.state import Violation

START = date(2026, 10, 10)
END = date(2026, 10, 13)
TODAY = date(2026, 9, 28)

FREE_PLAN = ItineraryDecision(
    assignments=[
        Assignment(day=2, slot="morning", activity_id="BCN-A04"),  # free walk, Sunday
        Assignment(day=3, slot="afternoon", activity_id="BCN-A07"),  # free beach, Monday
    ],
    day_themes=[],
)


def _request(budget):
    return TripRequest(origin="London", destination="Barcelona", start_date=START,
                       end_date=END, travellers=2, budget_amount=budget)


def _flight(selected):
    return FlightDecision(best_value_id=selected, selected_id=selected, reasons=[])


def _stay(selected):
    return StayDecision(best_value_id=selected, selected_id=selected, reasons=[])


ACTIVITIES = ActivityDecision(
    picks=[ActivityPick(id="BCN-A04", reason="Free walk."),
           ActivityPick(id="BCN-A07", reason="Free beach.")],
    note="Test note.",
)


def _agents(state):
    return [t.agent for t in state["trace"]]


# ---------- Happy path ----------
def test_valid_plan_first_time():
    fake = FakeLLM({
        FlightDecision: [_flight("BCN-F1")],
        ActivityDecision: [ACTIVITIES],
        StayDecision: [_stay("BCN-S1")],
        ItineraryDecision: [FREE_PLAN],
    })
    state = plan_trip(fake, request=_request(2000))

    assert state["status"] == "valid"
    assert state["revision_count"] == 0
    assert state["budget"].total == pytest.approx(1365.10)

    agents = _agents(state)
    assert agents[0] == "parser"
    assert set(agents[1:3]) == {"flight_agent", "activities_agent"}  # the parallel step
    assert agents[3:] == ["stay_agent", "itinerary_agent", "budget", "validate", "router", "compile"]
    assert sum(t.llm_calls for t in state["trace"]) == 4  # no advisor: within budget


# ---------- The revision loop ----------
def test_over_budget_is_fixed_by_a_cheaper_stay():
    fake = FakeLLM({
        FlightDecision: [_flight("BCN-F1")],
        ActivityDecision: [ACTIVITIES],
        StayDecision: [_stay("BCN-S1"), _stay("BCN-S2")],  # second answer is the revision
        ItineraryDecision: [FREE_PLAN, FREE_PLAN],
        AdvisorDecision: [AdvisorDecision(action="cheaper_stay", reason="Save on the hotel.")],
    })
    state = plan_trip(fake, request=_request(1200))

    assert state["status"] == "valid"
    assert state["revision_count"] == 1
    assert state["selected_stay_id"] == "BCN-S2"
    assert "BCN-S1" in state["excluded_ids"]
    assert state["budget"].total == pytest.approx(1078.00)
    assert state["budget"].over_by == 0

    router_summaries = [t.summary for t in state["trace"] if t.agent == "router"]
    assert router_summaries[0] == "Revision 1: over_budget sent to stay_agent (cheaper_stay)."
    assert router_summaries[-1] == "Plan is valid after 1 revision(s)."


def test_thin_plan_is_repaired_by_the_loop():
    thin = ItineraryDecision(
        assignments=[Assignment(day=2, slot="morning", activity_id="BCN-A04")],  # Monday empty
        day_themes=[],
    )
    fake = FakeLLM({
        FlightDecision: [_flight("BCN-F1")],
        ActivityDecision: [ACTIVITIES],
        StayDecision: [_stay("BCN-S1")],
        ItineraryDecision: [thin, FREE_PLAN],  # the revision fills Monday
    })
    state = plan_trip(fake, request=_request(2000))

    assert state["status"] == "valid"
    assert state["revision_count"] == 1
    router_summaries = [t.summary for t in state["trace"] if t.agent == "router"]
    assert router_summaries[0] == "Revision 1: thin_plan sent to itinerary_agent."


def test_unaffordable_budget_stops_cleanly_with_warnings():
    fake = FakeLLM({
        FlightDecision: [_flight("BCN-F1"), _flight("BCN-F3")],
        ActivityDecision: [ACTIVITIES],
        StayDecision: [_stay("BCN-S1"), _stay("BCN-S2"), _stay("BCN-S3")],
        ItineraryDecision: [FREE_PLAN] * 4,
        AdvisorDecision: [
            AdvisorDecision(action="cheaper_stay", reason="Cheaper hotel."),
            AdvisorDecision(action="cheaper_flight", reason="Cheaper flight."),
        ],
    })
    state = plan_trip(fake, request=_request(300))  # impossible for 2 people over 4 days

    assert state["status"] == "invalid"
    assert state["revision_count"] <= 3
    assert any("over the budget" in w for w in state["warnings"])
    assert _agents(state)[-1] == "compile"


def test_router_stops_after_max_revisions(barcelona_request):
    state = {
        "request": barcelona_request,
        "revision_count": 3,
        "violations": [Violation(code="over_budget", message="Plan is GBP 50.00 over.")],
        "trace": [],
    }
    update = router_node(state)
    assert update["status"] == "invalid"
    assert update["warnings"] == ["Plan is GBP 50.00 over."]
    assert "Stopped after 3 revisions" in update["trace"][0].summary


# ---------- Stopping early ----------
def test_vague_text_stops_with_a_question():
    unclear = ParsedRequest(
        origin=None, destination=None, start_date=None, end_date=None, duration_days=None,
        travellers=None, budget_amount=None, currency=None, style=None, pace=None, interests=[],
    )
    fake = FakeLLM({ParsedRequest: [unclear]})
    state = plan_trip(fake, raw_request="somewhere sunny", today=TODAY)

    assert state["status"] == "failed"
    assert state["warnings"] == ["Where would you like to go?"]
    assert _agents(state) == ["parser"]  # no agent ran


def test_unsupported_city_fails_cleanly():
    paris = TripRequest(origin="London", destination="Paris", start_date=START, end_date=END)
    fake = FakeLLM({})
    state = plan_trip(fake, request=paris)

    assert state["status"] == "failed"
    assert any("Demo data covers" in w for w in state["warnings"])
    assert fake.calls == []
    assert _agents(state)[-1] == "compile"