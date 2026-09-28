"""Deterministic plan validation.

Every plan passes through these checks before it is shown to the user.
Returns a list of violations; an empty list means the plan is valid.
No LLM is involved.
"""

from datetime import date, datetime, time, timedelta, tzinfo

from schemas.models import (
    WEEKDAYS,
    ActivityCandidate,
    BudgetBreakdown,
    DayPlan,
    FlightOption,
    StayOption,
    TripRequest,
)
from schemas.state import Violation, ViolationCode

# ---------- Rules ----------
ARRIVAL_BUFFER = timedelta(minutes=90)  # baggage and transfer after landing
DEPARTURE_BUFFER = timedelta(hours=3)  # transfer and airport time before the return flight
LATE_ARRIVAL_FROM = time(21, 0)  # landing at or after this counts as late
LATE_ARRIVAL_UNTIL = time(5, 0)  # ...and so does landing after midnight, before this

# Which node should fix each violation. None means re-planning cannot fix it.
VIOLATION_OWNER: dict[ViolationCode, str | None] = {
    "activity_outside_trip": "itinerary_agent",
    "activity_before_landing": "itinerary_agent",
    "activity_after_departure": "itinerary_agent",
    "activity_on_closed_day": "itinerary_agent",
    "activity_outside_hours": "itinerary_agent",
    "overlapping_items": "itinerary_agent",
    "empty_day": "itinerary_agent",
    "unknown_activity": "itinerary_agent",
    "stay_dates_mismatch": "stay_agent",
    "no_late_checkin": "stay_agent",
    "over_budget": "budget_advisor",
    "currency_mismatch": None,
}


# ---------- Helpers ----------
def _v(code: ViolationCode, message: str, item_ids: list[str] | None = None) -> Violation:
    return Violation(code=code, message=message, item_ids=item_ids or [])


def _at(day: date, clock: time, tz: tzinfo) -> datetime:
    return datetime.combine(day, clock, tzinfo=tz)


def _trip_dates(request: TripRequest) -> list[date]:
    return [request.start_date + timedelta(days=i) for i in range(request.num_days)]


def _item_ref(item) -> str:
    return item.activity_id or item.title


def is_late_arrival(flight: FlightOption) -> bool:
    """Shared rule: landing at 21:00 or later, or after midnight, counts as late.
    Used by the validator and the Stay Agent so they always agree."""
    landing = flight.outbound_arrive.time()
    return landing >= LATE_ARRIVAL_FROM or landing < LATE_ARRIVAL_UNTIL


# ---------- Individual checks ----------
def check_days(request: TripRequest, itinerary: list[DayPlan]) -> list[Violation]:
    violations = []
    trip_dates = set(_trip_dates(request))

    for day in itinerary:
        if day.date not in trip_dates:
            violations.append(
                _v("activity_outside_trip", f"{day.date} is outside the trip dates.",
                   [_item_ref(i) for i in day.items])
            )

    planned = {day.date for day in itinerary if day.items}
    for d in sorted(trip_dates - planned):
        violations.append(_v("empty_day", f"No plan for {d}."))
    return violations


def check_flight_window(flight: FlightOption | None, itinerary: list[DayPlan]) -> list[Violation]:
    if flight is None:
        return []
    tz = flight.outbound_arrive.tzinfo  # destination local time (provider contract)
    earliest = flight.outbound_arrive + ARRIVAL_BUFFER
    latest = flight.return_depart - DEPARTURE_BUFFER

    violations = []
    for day in itinerary:
        for item in day.items:
            if item.kind == "transfer" or item.start_time is None or item.end_time is None:
                continue  # airport transfers are allowed near flight times
            start = _at(day.date, item.start_time, tz)
            end = _at(day.date, item.end_time, tz)
            if start < earliest:
                violations.append(
                    _v("activity_before_landing",
                       f"'{item.title}' starts {start:%a %H:%M}, before landing plus buffer "
                       f"({earliest:%a %H:%M}).", [_item_ref(item)])
                )
            if end > latest:
                violations.append(
                    _v("activity_after_departure",
                       f"'{item.title}' ends {end:%a %H:%M}, too close to the return flight "
                       f"(latest {latest:%a %H:%M}).", [_item_ref(item)])
                )
    return violations


def check_activities(itinerary: list[DayPlan], activities: list[ActivityCandidate]) -> list[Violation]:
    by_id = {a.id: a for a in activities}
    violations = []
    for day in itinerary:
        weekday = WEEKDAYS[day.date.weekday()]
        for item in day.items:
            if item.kind != "activity":
                continue
            activity = by_id.get(item.activity_id)
            if activity is None:
                violations.append(
                    _v("unknown_activity",
                       f"'{item.activity_id}' is not in the candidate pool.", [item.activity_id])
                )
                continue
            if weekday in activity.closed_days:
                violations.append(
                    _v("activity_on_closed_day",
                       f"{activity.name} is closed on {weekday} ({day.date}).", [activity.id])
                )
            if (item.start_time and item.end_time and activity.open_time and activity.close_time
                    and (item.start_time < activity.open_time or item.end_time > activity.close_time)):
                violations.append(
                    _v("activity_outside_hours",
                       f"{activity.name} is scheduled {item.start_time:%H:%M}-{item.end_time:%H:%M} "
                       f"but opens {activity.open_time:%H:%M}-{activity.close_time:%H:%M}.",
                       [activity.id])
                )
    return violations


def check_overlaps(itinerary: list[DayPlan]) -> list[Violation]:
    violations = []
    for day in itinerary:
        timed = sorted(
            (i for i in day.items if i.start_time and i.end_time), key=lambda i: i.start_time
        )
        for prev, cur in zip(timed, timed[1:]):
            if cur.start_time < prev.end_time:
                violations.append(
                    _v("overlapping_items",
                       f"'{prev.title}' and '{cur.title}' overlap on {day.date}.",
                       [_item_ref(prev), _item_ref(cur)])
                )
    return violations


def check_stay(request: TripRequest, flight: FlightOption | None, stay: StayOption | None) -> list[Violation]:
    violations = []
    if request.num_nights > 0 and stay is None:
        violations.append(
            _v("stay_dates_mismatch", f"No accommodation selected for {request.num_nights} nights.")
        )
    if stay and flight and is_late_arrival(flight) and not stay.late_checkin_ok:
        violations.append(
            _v("no_late_checkin",
               f"Flight lands at {flight.outbound_arrive:%H:%M} but {stay.name} "
               "has no late check-in.",
               [stay.id, flight.id])
        )
    return violations


def check_budget(budget: BudgetBreakdown | None) -> list[Violation]:
    if budget and budget.over_by > 0:
        return [
            _v("over_budget",
               f"Plan is {budget.currency} {budget.over_by:,.2f} over the budget of "
               f"{budget.currency} {budget.budget_amount:,.2f}.")
        ]
    return []


def check_currency(
    request: TripRequest,
    flight: FlightOption | None,
    stay: StayOption | None,
    activities: list[ActivityCandidate],
) -> list[Violation]:
    priced = [x for x in [flight, stay, *activities] if x is not None]
    wrong = [x.id for x in priced if x.currency != request.currency]
    if wrong:
        return [_v("currency_mismatch", f"Items not priced in {request.currency}.", wrong)]
    return []


# ---------- Entry point ----------
def validate_plan(
    request: TripRequest,
    flight: FlightOption | None,
    stay: StayOption | None,
    itinerary: list[DayPlan],
    activities: list[ActivityCandidate],
    budget: BudgetBreakdown | None,
) -> list[Violation]:
    return [
        *check_currency(request, flight, stay, activities),
        *check_days(request, itinerary),
        *check_flight_window(flight, itinerary),
        *check_activities(itinerary, activities),
        *check_overlaps(itinerary),
        *check_stay(request, flight, stay),
        *check_budget(budget),
    ]