"""Flight Agent.

Code fetches options and computes the facts (cheapest, fastest, usable hours
at the destination, share of the budget). The LLM makes the judgement call
(best value, which to select, reasons). Code then checks the LLM's choice and
falls back to a rule if it is invalid.
"""

from datetime import timedelta

from pydantic import BaseModel

from agents.common import DATA_NOTICE, Timer, data_block, trace_entry
from agents.llm import LLM, LLMError
from core.scheduler import _day_window as day_window  # same rule the scheduler uses
from providers.base import FlightProvider, UnsupportedDestinationError
from schemas.models import FlightOption, TripRequest
from schemas.state import TripState

AGENT = "flight_agent"


# ---------- What the LLM is allowed to return ----------
class FlightReason(BaseModel):
    id: str
    reason: str


class FlightDecision(BaseModel):
    best_value_id: str
    selected_id: str
    reasons: list[FlightReason]


SYSTEM_PROMPT = f"""You are the Flight Agent in a travel planner.
Choose flights for the traveller from the options provided.
- best_value_id: the option with the best balance of price and usable time at the destination.
- selected_id: the option you recommend overall for this traveller.
- reasons: one short sentence per option explaining its trade-off.
Each option shows the usable hours at the destination (already computed from the
flight times) and, when there is a budget, the share of the budget the flights take.
Prefer options that give more usable time for less money. Only choose a more
expensive option when it clearly adds usable time, or the traveller's style is luxury.
Flights, accommodation, food and activities must all fit in one budget, so an
expensive flight means less money for everything else.
Only use IDs that appear in the options. Never invent flights, prices or times.
{DATA_NOTICE}"""


# ---------- Deterministic helpers ----------
def _total_mins(f: FlightOption) -> int:
    return f.outbound_duration_mins + f.return_duration_mins


def _cheapest(options: list[FlightOption]) -> FlightOption:
    return min(options, key=lambda f: (f.price_per_person, _total_mins(f)))


def _fastest(options: list[FlightOption]) -> FlightOption:
    return min(options, key=lambda f: (_total_mins(f), f.price_per_person))


def _fallback_pick(options: list[FlightOption]) -> FlightOption:
    """Used when the LLM fails: cheapest direct flight, else cheapest overall."""
    direct = [f for f in options if f.outbound_stops == 0 and f.return_stops == 0]
    return _cheapest(direct or options)


def usable_hours(request: TripRequest, flight: FlightOption) -> float:
    """Hours available for activities across the trip with this flight,
    using exactly the same day windows as the scheduler."""
    total_mins = 0
    for i in range(request.num_days):
        window = day_window(request.start_date + timedelta(days=i), flight)
        if window:
            total_mins += window[1] - window[0]
    return round(total_mins / 60, 1)


def _describe(f: FlightOption, request: TripRequest) -> str:
    flights_total = f.price_per_person * request.travellers
    share = (
        f" ({flights_total / request.budget_amount:.0%} of the budget)"
        if request.budget_amount else ""
    )
    return (
        f"{f.id} | {f.airline} | out {f.outbound_depart:%a %H:%M} -> "
        f"{f.outbound_arrive:%H:%M} local, {f.outbound_stops} stops | "
        f"back {f.return_depart:%a %H:%M} -> {f.return_arrive:%H:%M}, "
        f"{f.return_stops} stops | usable time at destination {usable_hours(request, f)} h | "
        f"{f.currency} {f.price_per_person:.2f} per person, "
        f"{f.currency} {flights_total:.2f} for {request.travellers}{share}"
    )


def _failed(timer: Timer, message: str, revision: int) -> dict:
    return {
        "flight_options": [],
        "selected_flight_id": None,
        "trace": [trace_entry(AGENT, "failed", timer, message, revision=revision)],
        "warnings": [message],
    }


# ---------- The node ----------
def run_flight_agent(state: TripState, provider: FlightProvider, llm: LLM) -> dict:
    timer = Timer()
    request = state["request"]
    revision = state.get("revision_count", 0)
    rev = state.get("revision_request")
    is_my_revision = rev is not None and rev.target == AGENT

    # 1. Fetch
    try:
        options = provider.search_flights(
            request.origin, request.destination, request.start_date, request.end_date
        )
    except UnsupportedDestinationError as error:
        return _failed(timer, str(error), revision)

    # 2. Filter
    excluded = set(state.get("excluded_ids", []))
    options = [o for o in options if o.id not in excluded]
    if is_my_revision and rev.budget_action == "cheaper_flight":
        current = next(
            (o for o in state.get("flight_options", []) if o.id == state.get("selected_flight_id")),
            None,
        )
        if current:
            options = [o for o in options if o.price_per_person < current.price_per_person]
    if not options:
        return _failed(timer, "No suitable flights found for these dates.", revision)

    # 3. Facts, computed by code
    cheapest, fastest = _cheapest(options), _fastest(options)
    by_id = {o.id: o for o in options}
    budget_line = (
        f"Whole-trip budget: {request.currency} {request.budget_amount:.2f}.\n"
        if request.budget_amount else "No budget given.\n"
    )

    # 4. Judgement, made by the LLM
    user_prompt = (
        f"Traveller: style={request.style}, pace={request.pace}, "
        f"travellers={request.travellers}, {request.num_days}-day trip.\n"
        + budget_line
        + f"Facts already computed: cheapest={cheapest.id}, fastest={fastest.id}.\n"
        + (f"Revision request: {rev.reason}\n" if is_my_revision else "")
        + "Options:\n"
        + data_block("\n".join(_describe(o, request) for o in options))
    )

    status, warnings, llm_result, reasons = "success", [], None, {}
    try:
        llm_result = llm.structured(SYSTEM_PROMPT, user_prompt, FlightDecision)
        decision = llm_result.parsed
        # 5. Check the decision
        unknown = {decision.best_value_id, decision.selected_id} - by_id.keys()
        if unknown:
            raise LLMError(f"LLM returned unknown flight IDs: {sorted(unknown)}")
        best_value_id, selected_id = decision.best_value_id, decision.selected_id
        reasons = {r.id: r.reason for r in decision.reasons if r.id in by_id}
    except LLMError as error:
        status = "partial"
        warnings.append(f"Used fallback rule (cheapest direct flight): {error}")
        best_value_id = selected_id = _fallback_pick(options).id

    # 6. Attach every label that applies, plus reasons, to the real provider data
    labels: dict[str, list[str]] = {}
    for option_id, label in [
        (cheapest.id, "cheapest"),
        (fastest.id, "fastest"),
        (best_value_id, "best_value"),
    ]:
        labels.setdefault(option_id, []).append(label)

    shortlist: list[FlightOption] = []
    for option_id in dict.fromkeys([selected_id, cheapest.id, fastest.id, best_value_id]):
        shortlist.append(
            by_id[option_id].model_copy(
                update={"labels": labels.get(option_id, []), "reason": reasons.get(option_id)}
            )
        )

    chosen = by_id[selected_id]
    summary = (
        f"{len(options)} options found; selected {chosen.id} ({chosen.airline}, "
        f"{chosen.currency} {chosen.price_per_person:.2f} per person, "
        f"lands {chosen.outbound_arrive:%H:%M}, "
        f"{usable_hours(request, chosen)} usable hours)."
    )
    return {
        "flight_options": shortlist,
        "selected_flight_id": selected_id,
        "trace": [
            trace_entry(AGENT, status, timer, summary, llm_result,
                        warnings=warnings, revision=revision)
        ],
    }