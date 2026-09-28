"""Stay Agent.

Runs after the Flight and Activities agents, because it needs the flight's
arrival time and where the activities are. Code applies hard filters and
computes facts (price, distance); the LLM weighs the trade-offs.
"""

import statistics

from pydantic import BaseModel

from agents.common import DATA_NOTICE, Timer, data_block, trace_entry
from agents.llm import LLM, LLMError
from core.budget import rooms_needed
from core.geo import mean_distance_km
from core.validator import is_late_arrival
from providers.base import StayProvider, UnsupportedDestinationError
from schemas.models import FlightOption, StayOption
from schemas.state import TripState

AGENT = "stay_agent"
TOP_ACTIVITIES = 6  # distance is measured to the best-ranked activity candidates


# ---------- What the LLM is allowed to return ----------
class StayReason(BaseModel):
    id: str
    reason: str


class StayDecision(BaseModel):
    best_value_id: str
    selected_id: str
    reasons: list[StayReason]


SYSTEM_PROMPT = f"""You are the Stay Agent in a travel planner.
Choose accommodation for the traveller from the options provided.
- best_value_id: the best balance of total price, location and rating.
- selected_id: the option you recommend overall for this traveller's style and budget.
- reasons: one short sentence per option explaining its trade-off
  (total price, distance to planned activities, rating).
Distances are the average straight-line km from the hotel to the traveller's top
activities; lower means less travel each day.
Only use IDs that appear in the options. Never invent hotels, prices or ratings.
{DATA_NOTICE}"""


# ---------- Deterministic helpers ----------
def _selected(options: list, selected_id: str | None):
    return next((o for o in options if o.id == selected_id), None)


def _describe(s: StayOption, nights: int, rooms: int, dist: float | None) -> str:
    total = s.total_price(nights) * rooms
    rating = f"{s.rating}" if s.rating is not None else "n/a"
    distance = f"avg {dist:.1f} km to activities" if dist is not None else "distance unknown"
    return (
        f"{s.id} | {s.name} | area: {s.area} | {s.currency} {s.price_per_night:.2f} per night, "
        f"{s.currency} {total:.2f} total for {nights} nights x {rooms} room(s) | "
        f"rating {rating} | {distance} | late check-in {'yes' if s.late_checkin_ok else 'no'}"
    )


def _fallback_pick(options: list[StayOption], dists: dict[str, float | None]) -> StayOption:
    """Used when the LLM fails: closest hotel at or below the median price."""
    median = statistics.median(s.price_per_night for s in options)
    affordable = [s for s in options if s.price_per_night <= median]
    return min(
        affordable,
        key=lambda s: (dists[s.id] if dists[s.id] is not None else float("inf"), s.price_per_night),
    )


def _failed(timer: Timer, message: str, revision: int) -> dict:
    return {
        "stay_options": [],
        "selected_stay_id": None,
        "trace": [trace_entry(AGENT, "failed", timer, message, revision=revision)],
        "warnings": [message],
    }


# ---------- The node ----------
def run_stay_agent(state: TripState, provider: StayProvider, llm: LLM) -> dict:
    timer = Timer()
    request = state["request"]
    revision = state.get("revision_count", 0)
    rev = state.get("revision_request")
    is_my_revision = rev is not None and rev.target == AGENT

    # 1. Same-day trip: nothing to do
    if request.num_nights == 0:
        return {
            "stay_options": [],
            "selected_stay_id": None,
            "trace": [trace_entry(AGENT, "success", timer,
                                  "Same-day trip; no accommodation needed.", revision=revision)],
        }

    # 2. Fetch
    try:
        options = provider.search_stays(request.destination, request.start_date, request.end_date)
    except UnsupportedDestinationError as error:
        return _failed(timer, str(error), revision)

    # 3. Hard filters (facts, not judgement)
    notes: list[str] = []
    excluded = set(state.get("excluded_ids", []))
    options = [s for s in options if s.id not in excluded]

    flight: FlightOption | None = _selected(
        state.get("flight_options", []), state.get("selected_flight_id")
    )
    if flight and is_late_arrival(flight):
        before = len(options)
        options = [s for s in options if s.late_checkin_ok]
        if len(options) < before:
            notes.append(
                f"Removed {before - len(options)} option(s) without late check-in "
                f"(flight lands {flight.outbound_arrive:%H:%M})."
            )

    if is_my_revision and rev.budget_action == "cheaper_stay":
        current = _selected(state.get("stay_options", []), state.get("selected_stay_id"))
        if current:
            options = [s for s in options if s.price_per_night < current.price_per_night]

    if not options:
        return _failed(timer, "No accommodation matches the trip's requirements.", revision)

    # 4. Facts, computed by code
    nights, rooms = request.num_nights, rooms_needed(request.travellers)
    top_activities = state.get("activity_candidates", [])[:TOP_ACTIVITIES]
    dists = {s.id: mean_distance_km(s, top_activities) for s in options}
    by_id = {s.id: s for s in options}

    cheapest = min(options, key=lambda s: (s.price_per_night, s.id))
    located = [s for s in options if dists[s.id] is not None]
    closest = min(located, key=lambda s: (dists[s.id], s.price_per_night)) if located else None
    if closest is None:
        notes.append("No activity locations available, so distance was not considered.")

    # 5. Judgement, made by the LLM
    budget_line = (
        f"Whole-trip budget: {request.currency} {request.budget_amount:.2f}.\n"
        if request.budget_amount else "No budget given.\n"
    )
    user_prompt = (
        f"Traveller: style={request.style}, pace={request.pace}, "
        f"travellers={request.travellers}, {nights} nights, {rooms} room(s).\n"
        + budget_line
        + f"Facts already computed: cheapest={cheapest.id}, "
        f"closest to activities={closest.id if closest else 'unknown'}.\n"
        + (f"Revision request: {rev.reason}\n" if is_my_revision else "")
        + "Options:\n"
        + data_block("\n".join(_describe(s, nights, rooms, dists[s.id]) for s in options))
    )

    status, warnings, llm_result, reasons = "success", [], None, {}
    try:
        llm_result = llm.structured(SYSTEM_PROMPT, user_prompt, StayDecision)
        decision = llm_result.parsed
        unknown = {decision.best_value_id, decision.selected_id} - by_id.keys()
        if unknown:
            raise LLMError(f"LLM returned unknown stay IDs: {sorted(unknown)}")
        best_value_id, selected_id = decision.best_value_id, decision.selected_id
        reasons = {r.id: r.reason for r in decision.reasons if r.id in by_id}
    except LLMError as error:
        status = "partial"
        warnings.append(f"Used fallback rule (closest at or below median price): {error}")
        best_value_id = selected_id = _fallback_pick(options, dists).id

    # 6. Attach every label that applies, plus reasons, to the real provider data
    labels: dict[str, list[str]] = {}
    labelled = [(cheapest.id, "cheapest"), (best_value_id, "best_value")]
    if closest:
        labelled.insert(1, (closest.id, "best_location"))
    for option_id, label in labelled:
        labels.setdefault(option_id, []).append(label)

    order = [selected_id, cheapest.id, closest.id if closest else None, best_value_id]
    shortlist = [
        by_id[i].model_copy(update={"labels": labels.get(i, []), "reason": reasons.get(i)})
        for i in dict.fromkeys(i for i in order if i)
    ]

    chosen = by_id[selected_id]
    distance = f", avg {dists[chosen.id]:.1f} km to activities" if dists[chosen.id] is not None else ""
    summary = (
        f"{len(options)} options considered; selected {chosen.name} ({chosen.area}, "
        f"{chosen.currency} {chosen.price_per_night:.2f} per night{distance})."
    )
    if notes:
        summary += " " + " ".join(notes)
    return {
        "stay_options": shortlist,
        "selected_stay_id": selected_id,
        "trace": [
            trace_entry(AGENT, status, timer, summary, llm_result,
                        warnings=warnings, revision=revision)
        ],
    }