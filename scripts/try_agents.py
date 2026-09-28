"""Manual smoke test: runs the four agents in order with the real LLM,
then costs and validates the plan. Not part of the test suite.

Usage (from the project root):  python -m scripts.try_agents
"""

from datetime import date

from agents.activities import run_activities_agent
from agents.flight import run_flight_agent
from agents.itinerary import run_itinerary_agent
from agents.llm import OpenAILLM
from agents.stay import run_stay_agent
from core.budget import compute_budget
from core.validator import validate_plan
from providers.mock import MockActivityProvider, MockFlightProvider, MockStayProvider
from schemas.models import TripRequest

APPEND_KEYS = {"trace", "warnings", "excluded_ids"}  # same as the reducers in TripState


def merge(state: dict, update: dict) -> None:
    """Apply a node's update the way LangGraph will: append some keys, replace the rest."""
    for key, value in update.items():
        state[key] = state.get(key, []) + value if key in APPEND_KEYS else value


def selected(options: list, selected_id):
    return next((o for o in options if o.id == selected_id), None)


def main() -> None:
    request = TripRequest(
        origin="London",
        destination="Barcelona",
        start_date=date(2026, 10, 10),
        end_date=date(2026, 10, 13),
        travellers=2,
        budget_amount=1500,
        interests=["architecture", "food", "nightlife"],
    )
    llm = OpenAILLM()
    state: dict = {"request": request}

    merge(state, run_flight_agent(state, MockFlightProvider(), llm))
    merge(state, run_activities_agent(state, MockActivityProvider(), llm))
    merge(state, run_stay_agent(state, MockStayProvider(), llm))
    merge(state, run_itinerary_agent(state, llm))

    print("\n=== TRACE ===")
    for t in state["trace"]:
        print(f"{t.agent:18} {t.status:8} {t.latency_ms:6} ms  {t.tokens:5} tok  {t.summary}")
        for w in t.warnings:
            print(f"{'':18} warning: {w}")

    print("\n=== ITINERARY ===")
    for day in state["itinerary"]:
        print(f"\n{day.date:%a %d %b}  {day.notes or ''}")
        for i in day.items:
            when = f"{i.start_time:%H:%M}-{i.end_time:%H:%M}" if i.start_time else "           "
            print(f"  {when}  {i.kind:9} {i.title}")

    flight = selected(state["flight_options"], state["selected_flight_id"])
    stay = selected(state["stay_options"], state["selected_stay_id"])
    budget = compute_budget(request, flight, stay, state["itinerary"], state["activity_candidates"])
    violations = validate_plan(
        request, flight, stay, state["itinerary"], state["activity_candidates"], budget
    )

    print("\n=== BUDGET ===")
    print(f"Total {budget.currency} {budget.total:,.2f} (per traveller {budget.per_traveller:,.2f}), "
          f"over by {budget.over_by:,.2f}")
    print("\n=== VALIDATOR ===")
    print([v.code for v in violations] or "No violations")


if __name__ == "__main__":
    main()