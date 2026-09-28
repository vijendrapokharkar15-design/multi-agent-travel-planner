"""Deterministic scheduler.

Turns the LLM's assignments ("activity X on day 2, afternoon") into a real
timetable with clock times, travel gaps, meals and airport transfers.
The LLM never writes times; this file does, using fixed, documented rules.
Anything that cannot fit is dropped with a warning instead of producing an
invalid plan.
"""

from datetime import date, datetime, time, timedelta

from pydantic import BaseModel

from core.geo import HasLocation, distance_km, travel_minutes
from core.validator import ARRIVAL_BUFFER, DEPARTURE_BUFFER
from schemas.models import (
    WEEKDAYS,
    ActivityCandidate,
    DayPlan,
    FlightOption,
    ItineraryItem,
    Slot,
    StayOption,
    TripRequest,
)


# ---------- Rules (minutes since midnight) ----------
def _m(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


DAY_START = _m("09:30")
DAY_END = _m("23:30")
MORNING_END = _m("13:00")
LUNCH = (_m("13:00"), _m("14:00"))
AFTERNOON = (_m("14:15"), _m("18:30"))
DINNER_BEFORE_EVENING = (_m("19:00"), _m("20:00"))  # when the evening activity isn't food
DINNER_ONLY = (_m("19:30"), _m("20:30"))  # when nothing is planned for the evening
EVENING_START_WITH_FOOD = _m("19:00")
TRANSFER_MINS = 60
LAST_MINUTE = _m("23:59")
ROUND_TO = 5  # activity start times are rounded up to the next 5 minutes

SLOTS: tuple[Slot, ...] = ("morning", "afternoon", "evening")


# ---------- Data in and out ----------
class Assignment(BaseModel):
    """One LLM decision. Also used as part of the LLM's output schema."""

    day: int  # 1-based day number within the trip
    slot: Slot
    activity_id: str


class ScheduleResult(BaseModel):
    days: list[DayPlan]
    scheduled_ids: list[str]
    warnings: list[str]


# ---------- Small helpers ----------
def _clock(minutes: int) -> time:
    return time(minutes // 60, minutes % 60)


def _minutes(moment: datetime) -> int:
    return moment.hour * 60 + moment.minute


def _round_up(minutes: int) -> int:
    """11:37 becomes 11:40. Rounding up can never cause an overlap or an early start."""
    return -(-minutes // ROUND_TO) * ROUND_TO


def _slot_for(minutes: int) -> Slot:
    if minutes < MORNING_END:
        return "morning"
    if minutes < AFTERNOON[1]:
        return "afternoon"
    return "evening"


def _fixed(kind: str, title: str, start: int, end: int) -> ItineraryItem:
    return ItineraryItem(
        kind=kind, title=title, slot=_slot_for(start),
        start_time=_clock(start), end_time=_clock(end),
    )


def _day_window(day: date, flight: FlightOption | None) -> tuple[int, int] | None:
    """Usable time for activities on this day, or None if there is none."""
    start, end = DAY_START, DAY_END
    if flight:
        ready = flight.outbound_arrive + ARRIVAL_BUFFER
        if day < ready.date():
            return None
        if day == ready.date():
            start = max(start, _minutes(ready))
        latest = flight.return_depart - DEPARTURE_BUFFER
        if day > latest.date():
            return None
        if day == latest.date():
            end = min(end, _minutes(latest))
    return (start, end) if start < end else None


def _arrival_transfer(day: date, flight: FlightOption | None) -> ItineraryItem | None:
    if flight and flight.outbound_arrive.date() == day:
        start = min(_minutes(flight.outbound_arrive) + 15, LAST_MINUTE - 1)
        return _fixed("transfer", "Airport to hotel", start, min(start + TRANSFER_MINS, LAST_MINUTE))
    return None


def _departure_transfer(day: date, flight: FlightOption | None) -> ItineraryItem | None:
    if flight:
        leave = flight.return_depart - DEPARTURE_BUFFER
        if leave.date() == day:
            start = _minutes(leave)
            return _fixed("transfer", "Hotel to airport", start, min(start + TRANSFER_MINS, LAST_MINUTE))
    return None


def _try_place(
    activity: ActivityCandidate,
    segment: tuple[int, int],
    cursor: int,
    here: HasLocation | None,
) -> tuple[int, int] | None:
    """Earliest start that fits the segment, opening hours and travel time."""
    travel = travel_minutes(distance_km(here, activity)) if here else 0
    start = max(segment[0], cursor + travel)
    if activity.open_time:
        start = max(start, _minutes(datetime.combine(date.min, activity.open_time)))
    start = _round_up(start)
    end = start + activity.typical_duration_mins
    limit = segment[1]
    if activity.close_time:
        limit = min(limit, _minutes(datetime.combine(date.min, activity.close_time)))
    return (start, end) if end <= limit else None


# ---------- Planning one day ----------
def _plan_day(
    window: tuple[int, int],
    queues: dict[Slot, list[ActivityCandidate]],
    stay: StayOption | None,
) -> tuple[list[ItineraryItem], list[str], list[ActivityCandidate]]:
    win_start, win_end = window
    items: list[ItineraryItem] = []
    placed: list[str] = []
    cursor, here = win_start, stay

    def fill(segment: tuple[int, int], queue: list[ActivityCandidate]) -> list[ActivityCandidate]:
        nonlocal cursor, here
        leftover = []
        for activity in queue:
            slot_time = _try_place(activity, segment, cursor, here) if segment[0] < segment[1] else None
            if slot_time is None:
                leftover.append(activity)
                continue
            start, end = slot_time
            items.append(ItineraryItem(
                kind="activity", title=activity.name, slot=_slot_for(start),
                activity_id=activity.id, start_time=_clock(start), end_time=_clock(end),
            ))
            placed.append(activity.id)
            cursor, here = end, activity
        return leftover

    def covers(block: tuple[int, int]) -> bool:
        return win_start <= block[0] and win_end >= block[1]

    # Morning, then lunch
    carry = fill((max(win_start, DAY_START), min(MORNING_END, win_end)), queues.get("morning", []))
    if covers(LUNCH):
        items.append(_fixed("meal", "Lunch", *LUNCH))
        cursor = max(cursor, LUNCH[1])

    # Afternoon (anything that didn't fit in the morning gets a second chance)
    carry = fill((max(win_start, AFTERNOON[0]), min(AFTERNOON[1], win_end)),
                 carry + queues.get("afternoon", []))

    # Dinner and evening
    evening = carry + queues.get("evening", [])
    evening_start = EVENING_START_WITH_FOOD
    if evening and not any(a.category == "food" for a in evening) and covers(DINNER_BEFORE_EVENING):
        items.append(_fixed("meal", "Dinner", *DINNER_BEFORE_EVENING))
        evening_start = DINNER_BEFORE_EVENING[1]
        cursor = max(cursor, evening_start)
    elif not evening and covers(DINNER_ONLY):
        items.append(_fixed("meal", "Dinner", *DINNER_ONLY))
    dropped = fill((max(win_start, evening_start), min(DAY_END, win_end)), evening)

    return items, placed, dropped


# ---------- Entry point ----------
def build_schedule(
    request: TripRequest,
    flight: FlightOption | None,
    stay: StayOption | None,
    activities: list[ActivityCandidate],
    assignments: list[Assignment],
) -> ScheduleResult:
    by_id = {a.id: a for a in activities}
    dates = [request.start_date + timedelta(days=i) for i in range(request.num_days)]
    warnings: list[str] = []
    used: set[str] = set()
    queues: dict[int, dict[Slot, list[ActivityCandidate]]] = {}

    # 1. Check every assignment before timing anything
    for a in assignments:
        if not 1 <= a.day <= len(dates):
            warnings.append(f"Ignored {a.activity_id}: day {a.day} is outside the trip.")
            continue
        activity = by_id.get(a.activity_id)
        if activity is None:
            warnings.append(f"Ignored unknown activity '{a.activity_id}'.")
            continue
        if activity.id in used:
            warnings.append(f"Ignored duplicate booking of {activity.name}.")
            continue
        day = dates[a.day - 1]
        if WEEKDAYS[day.weekday()] in activity.closed_days:
            warnings.append(f"{activity.name} is closed on day {a.day} ({day:%a}), so it was left out.")
            continue
        used.add(activity.id)
        queues.setdefault(a.day, {}).setdefault(a.slot, []).append(activity)

    # 2. Build each day
    days: list[DayPlan] = []
    scheduled: list[str] = []
    for number, day in enumerate(dates, start=1):
        items: list[ItineraryItem] = []
        arrival = _arrival_transfer(day, flight)
        if arrival:
            items.append(arrival)

        window = _day_window(day, flight)
        day_queues = queues.get(number, {})
        if window is None:
            dropped = [a for q in day_queues.values() for a in q]
        else:
            day_items, placed, dropped = _plan_day(window, day_queues, stay)
            items += day_items
            scheduled += placed
        for activity in dropped:
            warnings.append(f"{activity.name} did not fit on day {number}, so it was left out.")

        departure = _departure_transfer(day, flight)
        if departure:
            items.append(departure)
        if not items:
            items.append(ItineraryItem(kind="rest", title="Free time", slot="morning"))

        items.sort(key=lambda i: (i.start_time is not None, i.start_time or time(0)))
        days.append(DayPlan(date=day, items=items))

    return ScheduleResult(days=days, scheduled_ids=scheduled, warnings=warnings)