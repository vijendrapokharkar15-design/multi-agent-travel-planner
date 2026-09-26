"""Deterministic trip cost calculation.

Food and transport figures are documented estimates, not live prices.
No LLM is involved here: the same inputs always give the same result.
"""

import math

from schemas.models import (
    ActivityCandidate,
    BudgetBreakdown,
    DayPlan,
    FlightOption,
    StayOption,
    Style,
    TripRequest,
)

# ---------- Estimate rules (GBP, per person) ----------
ESTIMATE_CURRENCY = "GBP"
FOOD_PER_DAY: dict[Style, float] = {"budget": 30, "mid-range": 50, "luxury": 90}
TRANSPORT_PER_DAY: dict[Style, float] = {"budget": 6, "mid-range": 10, "luxury": 25}
AIRPORT_TRANSFER_EACH_WAY: dict[Style, float] = {"budget": 5, "mid-range": 12, "luxury": 40}
CONTINGENCY_RATE = 0.10
PEOPLE_PER_ROOM = 2


class CurrencyMismatchError(ValueError):
    """Raised instead of silently adding amounts in different currencies."""


# ---------- Helpers ----------
def rooms_needed(travellers: int) -> int:
    return math.ceil(travellers / PEOPLE_PER_ROOM)


def scheduled_activity_ids(itinerary: list[DayPlan]) -> list[str]:
    return [
        item.activity_id
        for day in itinerary
        for item in day.items
        if item.kind == "activity" and item.activity_id
    ]


def _check_currency(
    currency: str,
    flight: FlightOption | None,
    stay: StayOption | None,
    activities: list[ActivityCandidate],
) -> None:
    if currency != ESTIMATE_CURRENCY:
        raise CurrencyMismatchError(
            f"The MVP estimates costs in {ESTIMATE_CURRENCY} only, not {currency}."
        )
    priced = [x for x in [flight, stay, *activities] if x is not None]
    for item in priced:
        if item.currency != currency:
            raise CurrencyMismatchError(
                f"{item.id} is priced in {item.currency}, expected {currency}."
            )


# ---------- Main function ----------
def compute_budget(
    request: TripRequest,
    flight: FlightOption | None,
    stay: StayOption | None,
    itinerary: list[DayPlan],
    activities: list[ActivityCandidate],
) -> BudgetBreakdown:
    """Cost the whole trip. `activities` is the candidate pool; only the
    activities actually scheduled in `itinerary` are charged."""
    currency = request.currency
    _check_currency(currency, flight, stay, activities)

    people = request.travellers
    days = request.num_days
    style = request.style

    flights_cost = round(flight.price_per_person * people, 2) if flight else 0.0
    stay_cost = (
        round(stay.total_price(request.num_nights) * rooms_needed(people), 2) if stay else 0.0
    )

    by_id = {a.id: a for a in activities}
    activities_cost = 0.0
    for activity_id in scheduled_activity_ids(itinerary):
        if activity_id not in by_id:
            raise ValueError(f"Itinerary references unknown activity '{activity_id}'.")
        activities_cost += by_id[activity_id].price_per_person * people
    activities_cost = round(activities_cost, 2)

    food_cost = round(FOOD_PER_DAY[style] * days * people, 2)
    transport_cost = round(
        (TRANSPORT_PER_DAY[style] * days + AIRPORT_TRANSFER_EACH_WAY[style] * 2) * people, 2
    )

    subtotal = flights_cost + stay_cost + activities_cost + food_cost + transport_cost
    contingency = round(subtotal * CONTINGENCY_RATE, 2)
    total = round(subtotal + contingency, 2)

    # Compare against the budget only for the parts the user said it covers.
    over_by = 0.0
    if request.budget_amount is not None:
        counted = total
        if not request.budget_includes_flights:
            counted -= flights_cost
        if not request.budget_includes_stay:
            counted -= stay_cost
        over_by = max(0.0, round(counted - request.budget_amount, 2))

    return BudgetBreakdown(
        currency=currency,
        flights=flights_cost,
        stay=stay_cost,
        activities=activities_cost,
        food_estimate=food_cost,
        local_transport_estimate=transport_cost,
        contingency=contingency,
        total=total,
        per_traveller=round(total / people, 2),
        budget_amount=request.budget_amount,
        over_by=over_by,
    )