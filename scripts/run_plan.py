"""Run one real planning request through the whole graph and print the result.
Also saves the full final state as JSON in runs/ (ignored by git).

Usage (from the project root):
  python -m scripts.run_plan                      # the Barcelona demo trip
  python -m scripts.run_plan "4 days in Lisbon from 10 Oct for 2, £1500, food"
"""

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

from agents.llm import OpenAILLM
from graph.build import plan_trip
from schemas.models import TripRequest

RUNS_DIR = Path("runs")

DEMO_REQUEST = TripRequest(
    origin="London",
    destination="Barcelona",
    start_date=date(2026, 10, 10),
    end_date=date(2026, 10, 13),
    travellers=2,
    budget_amount=1500,
    interests=["architecture", "food", "nightlife"],
)


def to_json(value):
    """Lets json.dumps handle Pydantic models and dates."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def selected(options, selected_id):
    return next((o for o in options or [] if o.id == selected_id), None)


def main() -> None:
    raw = " ".join(sys.argv[1:]).strip()
    started = time.perf_counter()
    if raw:
        state = plan_trip(OpenAILLM(), raw_request=raw)
    else:
        state = plan_trip(OpenAILLM(), request=DEMO_REQUEST)
    seconds = time.perf_counter() - started

    print("\n=== TRACE ===")
    for t in state["trace"]:
        print(f"[rev {t.revision}] {t.agent:17} {t.status:8} {t.latency_ms:6} ms  {t.summary}")
        for w in t.warnings:
            print(f"{'':27}warning: {w}")

    if state.get("itinerary"):
        flight = selected(state.get("flight_options"), state.get("selected_flight_id"))
        stay = selected(state.get("stay_options"), state.get("selected_stay_id"))
        print("\n=== CHOICES ===")
        if flight:
            print(f"Flight: {flight.id} {flight.airline}, lands {flight.outbound_arrive:%H:%M}, "
                  f"GBP {flight.price_per_person:.2f} pp  {flight.labels}")
        if stay:
            print(f"Stay:   {stay.name} ({stay.area}), GBP {stay.price_per_night:.2f}/night  {stay.labels}")

        print("\n=== ITINERARY ===")
        for day in state["itinerary"]:
            print(f"\n{day.date:%a %d %b}  {day.notes or ''}")
            for i in day.items:
                when = f"{i.start_time:%H:%M}-{i.end_time:%H:%M}" if i.start_time else " " * 11
                print(f"  {when}  {i.kind:9} {i.title}")

    budget = state.get("budget")
    if budget:
        print(f"\n=== BUDGET ===\nTotal GBP {budget.total:,.2f} "
              f"(per traveller {budget.per_traveller:,.2f}), over by {budget.over_by:,.2f}")

    print(f"\n=== RESULT ===\nStatus: {state.get('status')} after "
          f"{state.get('revision_count', 0)} revision(s), {seconds:.1f} s")
    llm_calls = sum(t.llm_calls for t in state["trace"])
    cost = sum(t.cost_usd for t in state["trace"])
    print(f"LLM calls: {llm_calls}, estimated cost: ${cost:.4f}")
    for w in state.get("warnings", []):
        print(f"Warning: {w}")

    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(state, default=to_json, indent=2), encoding="utf-8")
    print(f"Saved full state to {path}")


if __name__ == "__main__":
    main()