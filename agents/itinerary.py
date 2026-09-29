"""Itinerary Agent.

Part A of the planning split: the LLM decides WHAT goes WHERE
(activity -> day and slot, plus a theme per day). Part B, core/scheduler.py,
decides exactly WHEN. The LLM never writes clock times.
"""

from datetime import timedelta

from pydantic import BaseModel

from agents.activities import PER_DAY
from agents.common import DATA_NOTICE, Timer, data_block, trace_entry
from agents.llm import LLM, LLMError
from core.budget import CONTINGENCY_RATE
from core.scheduler import Assignment, ScheduleResult, build_schedule
from core.scheduler import _day_window as day_window  # shared rule: usable time per day
from schemas.models import WEEKDAYS, ActivityCandidate, FlightOption, TripRequest
from schemas.state import TripState

AGENT = "itinerary_agent"


# ---------- What the LLM is allowed to return ----------
class DayTheme(BaseModel):
    day: int
    theme: str


class ItineraryDecision(BaseModel):
    assignments: list[Assignment]
    day_themes: list[DayTheme]


SYSTEM_PROMPT = f"""You are the Itinerary Agent in a travel planner.
Assign activities from the candidates to trip days and slots (morning, afternoon, evening).
Rules:
- Only use days that have usable time, and respect each day's usable window
  (an evening slot is no use on a day that ends at 16:45).
- Every day with a full usable window should have at least one activity.
- Never assign an activity on one of its closed days.
- Aim for the number of activities per full day given by the traveller's pace.
- Group activities in the same or nearby areas on the same day to reduce travel;
  the hotel is the daily base.
- Put nightlife and evening dining in the evening slot; put places with opening
  hours in a slot when they are open.
- Use each activity at most once. You do not have to use every candidate.
- day_themes: a short theme for each day that has activities (e.g. 'Gaudi and Eixample').
Do not write clock times; a scheduler sets them.
Only use IDs that appear in the candidates.
{DATA_NOTICE}"""


# ---------- Deterministic helpers ----------
def _selected(options: list, selected_id: str | None):
    return next((o for o in options if o.id == selected_id), None)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _trip_dates(request: TripRequest):
    return [request.start_date + timedelta(days=i) for i in range(request.num_days)]


def _day_lines(request: TripRequest, flight: FlightOption | None) -> str:
    lines = []
    for number, day in enumerate(_trip_dates(request), start=1):
        window = day_window(day, flight)
        usable = (
            f"usable {_hhmm(window[0])}-{_hhmm(window[1])}" if window
            else "no usable time (travel day)"
        )
        lines.append(f"Day {number} | {day:%a %d %b} | {usable}")
    return "\n".join(lines)


def _describe(a: ActivityCandidate) -> str:
    hours = (
        f"{a.open_time:%H:%M}-{a.close_time:%H:%M}" if a.open_time and a.close_time
        else "always open"
    )
    closed = ", ".join(a.closed_days) or "none"
    return (
        f"{a.id} | {a.name} | area: {a.area} | {a.category} | {a.typical_duration_mins} mins | "
        f"hours {hours} | closed: {closed} | {a.currency} {a.price_per_person:.2f} per person"
    )


def fallback_assignments(
    request: TripRequest,
    flight: FlightOption | None,
    pool: list[ActivityCandidate],
    cheap_first: bool = False,
) -> list[Assignment]:
    """Used when the LLM fails: spread the pool across usable days in turn,
    respecting closed days and the pace limit. The scheduler then times it."""
    dates = _trip_dates(request)
    usable = [n for n, d in enumerate(dates, start=1) if day_window(d, flight)]
    if not usable:
        return []
    cap = PER_DAY[request.pace]
    counts = {n: 0 for n in usable}
    ordered = sorted(pool, key=lambda a: a.price_per_person) if cheap_first else pool

    assignments, turn = [], 0
    for activity in ordered:
        for step in range(len(usable)):
            n = usable[(turn + step) % len(usable)]
            if counts[n] < cap and WEEKDAYS[dates[n - 1].weekday()] not in activity.closed_days:
                if activity.category in ("nightlife", "food"):
                    slot = "evening"
                else:
                    slot = "morning" if counts[n] % 2 == 0 else "afternoon"
                assignments.append(Assignment(day=n, slot=slot, activity_id=activity.id))
                counts[n] += 1
                turn = (turn + step + 1) % len(usable)
                break
    return assignments


def _cost(ids: list[str], by_id: dict[str, ActivityCandidate]) -> float:
    return sum(by_id[i].price_per_person for i in ids if i in by_id)


def _activities_per_day(result: ScheduleResult) -> dict[str, int]:
    """For each scheduled activity, how many activities its day has in total."""
    counts: dict[str, int] = {}
    for day in result.days:
        ids = [i.activity_id for i in day.items if i.kind == "activity"]
        for activity_id in ids:
            counts[activity_id] = len(ids)
    return counts


# ---------- The node ----------
def run_itinerary_agent(state: TripState, llm: LLM) -> dict:
    timer = Timer()
    request = state["request"]
    revision = state.get("revision_count", 0)
    rev = state.get("revision_request")
    is_my_revision = rev is not None and rev.target == AGENT
    drop_paid = is_my_revision and rev.budget_action == "drop_paid_activities"

    flight = _selected(state.get("flight_options", []), state.get("selected_flight_id"))
    stay = _selected(state.get("stay_options", []), state.get("selected_stay_id"))
    pool = state.get("activity_candidates", [])
    by_id = {a.id: a for a in pool}

    previous_ids = [
        i.activity_id for d in state.get("itinerary", []) for i in d.items if i.kind == "activity"
    ]

    # For a budget fix: how much activity spending to cut, per person
    advice = state.get("budget_advice")
    target_pp = 0.0
    if drop_paid and advice and advice.estimated_saving:
        target_pp = round(
            advice.estimated_saving / (request.travellers * (1 + CONTINGENCY_RATE)), 2
        )

    status, warnings, llm_result = "success", [], None
    themes: dict[int, str] = {}

    def schedule(assignments: list[Assignment]) -> ScheduleResult:
        return build_schedule(request, flight, stay, pool, assignments)

    # 1. LLM assigns activities to days and slots
    if not pool:
        status = "partial"
        warnings.append("No activity candidates, so the plan has travel and meals only.")
        assignments: list[Assignment] = []
    else:
        if drop_paid and target_pp:
            budget_line = (
                f"Budget fix: prefer free activities. Reduce activity spending by about "
                f"{request.currency} {target_pp:.2f} per person. Swap paid activities for free "
                "candidates rather than leaving days empty, and keep at least one activity "
                "on every full day.\n"
            )
        elif drop_paid:
            budget_line = "Budget fix: prefer free activities and use fewer paid ones than before.\n"
        else:
            budget_line = ""
        user_prompt = (
            f"Traveller: pace={request.pace} (about {PER_DAY[request.pace]} activities per full day), "
            f"style={request.style}, travellers={request.travellers}.\n"
            f"Hotel area: {stay.area if stay else 'no hotel (same-day trip)'}.\n"
            + (f"Revision request: {rev.reason}\n" if is_my_revision else "")
            + budget_line
            + "Days:\n" + _day_lines(request, flight) + "\n"
            + "Candidates:\n" + data_block("\n".join(_describe(a) for a in pool))
        )
        try:
            llm_result = llm.structured(SYSTEM_PROMPT, user_prompt, ItineraryDecision)
            assignments = llm_result.parsed.assignments
            themes = {t.day: t.theme for t in llm_result.parsed.day_themes}
        except LLMError as error:
            status = "partial"
            warnings.append(f"Used fallback round-robin plan: {error}")
            assignments = fallback_assignments(request, flight, pool, cheap_first=drop_paid)

    # 2. Code sets the times
    result = schedule(assignments)

    # If the LLM's plan scheduled nothing at all, fall back rather than show an empty trip
    if pool and not result.scheduled_ids and status == "success":
        status = "partial"
        warnings.append("LLM plan scheduled no activities; used fallback round-robin plan.")
        assignments = fallback_assignments(request, flight, pool, cheap_first=drop_paid)
        themes = {}
        result = schedule(assignments)

    # 3. Budget fix: code makes sure the saving really happens, removing paid
    #    activities from BUSY days first so no day is emptied if avoidable
    if drop_paid and previous_ids:
        goal = _cost(previous_ids, by_id) - (target_pp or 0.01)
        while _cost(result.scheduled_ids, by_id) > goal:
            paid = [i for i in result.scheduled_ids if by_id[i].price_per_person > 0]
            if not paid:
                break
            per_day = _activities_per_day(result)
            busy = [i for i in paid if per_day.get(i, 0) >= 2]
            victim = max(busy or paid, key=lambda i: by_id[i].price_per_person)
            warnings.append(f"Removed {by_id[victim].name} to reduce cost.")
            assignments = [a for a in assignments if a.activity_id != victim]
            result = schedule(assignments)

    days = [
        day.model_copy(update={"notes": themes.get(number)})
        for number, day in enumerate(result.days, start=1)
    ]
    left_out = len([w for w in result.warnings if "left out" in w or "Ignored" in w])
    summary = (
        f"{len(result.scheduled_ids)} activities scheduled over {len(days)} days"
        + (f"; {left_out} left out." if left_out else ".")
    )
    return {
        "itinerary": days,
        "trace": [
            trace_entry(AGENT, status, timer, summary, llm_result,
                        warnings=warnings + result.warnings, revision=revision)
        ],
    }