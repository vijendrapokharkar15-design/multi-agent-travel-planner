"""Scheduler tests (pure code, no LLM).

Trip used throughout: London to Barcelona, Sat 10 to Tue 13 October 2026,
flight BCN-F1 (lands 10:15, returns 19:45), hotel BCN-S1 in the Gothic Quarter.
"""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from core.scheduler import Assignment, build_schedule
from core.validator import validate_plan
from providers.mock import MockFlightProvider, MockStayProvider

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


def _a(day, slot, activity_id):
    return Assignment(day=day, slot=slot, activity_id=activity_id)


def _times(day_plan):
    """[(title, 'HH:MM', 'HH:MM'), ...] for easy comparison."""
    return [
        (i.title, f"{i.start_time:%H:%M}", f"{i.end_time:%H:%M}")
        for i in day_plan.items if i.start_time
    ]


# ---------- The worked example ----------
def test_example_day_has_expected_times(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    result = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, [
        _a(2, "morning", "BCN-A01"),
        _a(2, "morning", "BCN-A02"),
        _a(2, "evening", "BCN-A12"),
    ])
    assert _times(result.days[1]) == [
        ("Sagrada Familia", "09:50", "11:20"),  # 20 min travel from the hotel
        ("Casa Batllo", "11:40", "12:55"),      # 17 min travel, rounded up to 11:40
        ("Lunch", "13:00", "14:00"),
        ("Dinner", "19:00", "20:00"),           # flamenco isn't food, so dinner comes first
        ("Flamenco show", "20:20", "21:50"),
    ]
    assert result.warnings == []


# ---------- The most important property ----------
def test_scheduler_output_passes_the_validator(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    result = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, [
        _a(1, "afternoon", "BCN-A04"),
        _a(1, "evening", "BCN-A08"),
        _a(2, "morning", "BCN-A01"),
        _a(2, "morning", "BCN-A02"),
        _a(2, "evening", "BCN-A12"),
        _a(3, "morning", "BCN-A11"),
        _a(3, "afternoon", "BCN-A09"),
        _a(4, "morning", "BCN-A06"),
    ])
    assert result.warnings == []
    violations = validate_plan(
        barcelona_request, bcn_f1, bcn_s1, result.days, bcn_activities, None
    )
    assert violations == []
    # every activity starts on a 5-minute boundary
    assert all(
        i.start_time.minute % 5 == 0
        for d in result.days for i in d.items if i.kind == "activity"
    )


def test_late_flight_plan_also_passes_the_validator(
    barcelona_request, bcn_late_flight, bcn_s1, bcn_activities
):
    # Lands 23:35, returns 06:30: no usable time on the first or last day
    result = build_schedule(barcelona_request, bcn_late_flight, bcn_s1, bcn_activities, [
        _a(1, "evening", "BCN-A12"),
        _a(2, "morning", "BCN-A01"),
    ])
    assert _times(result.days[0]) == [("Airport to hotel", "23:50", "23:59")]
    assert "Flamenco show did not fit on day 1, so it was left out." in result.warnings
    assert validate_plan(
        barcelona_request, bcn_late_flight, bcn_s1, result.days, bcn_activities, None
    ) == []


# ---------- Rejected assignments ----------
def test_closed_day_assignment_is_rejected(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    result = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, [
        _a(3, "morning", "BCN-A03"),  # Picasso Museum, closed on Mondays
    ])
    assert result.warnings == ["Picasso Museum is closed on day 3 (Mon), so it was left out."]
    assert "BCN-A03" not in result.scheduled_ids


def test_bad_assignments_are_ignored_with_warnings(
    barcelona_request, bcn_f1, bcn_s1, bcn_activities
):
    result = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, [
        _a(2, "morning", "BCN-A99"),  # invented by the LLM
        _a(9, "morning", "BCN-A01"),  # day outside the trip
        _a(2, "morning", "BCN-A02"),
        _a(3, "morning", "BCN-A02"),  # duplicate
    ])
    assert result.scheduled_ids == ["BCN-A02"]
    assert len(result.warnings) == 3


# ---------- Fitting rules ----------
def test_overflow_moves_to_the_afternoon(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    result = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, [
        _a(2, "morning", "BCN-A01"),
        _a(2, "morning", "BCN-A02"),
        _a(2, "morning", "BCN-A11"),  # no room left in the morning
    ])
    park = next(i for i in result.days[1].items if i.activity_id == "BCN-A11")
    assert park.slot == "afternoon"
    assert park.start_time >= time(14, 15)
    assert result.warnings == []


def test_activity_that_cannot_fit_anywhere_is_dropped(
    barcelona_request, bcn_f1, bcn_s1, bcn_activities
):
    # Last day ends at 16:45 (3 hours before the 19:45 flight), so no evening at all
    result = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, [
        _a(4, "evening", "BCN-A12"),
    ])
    assert "BCN-A12" not in result.scheduled_ids
    assert result.warnings == ["Flamenco show did not fit on day 4, so it was left out."]


def test_food_in_the_evening_replaces_dinner(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    result = build_schedule(barcelona_request, bcn_f1, bcn_s1, bcn_activities, [
        _a(2, "evening", "BCN-A08"),  # tapas evening, opens 19:30
    ])
    day2 = _times(result.days[1])
    assert ("Tapas evening in El Born", "19:30", "21:30") in day2
    assert not any(title == "Dinner" for title, _, _ in day2)


def test_day_with_no_usable_time_gets_free_time(barcelona_request, bcn_f1, bcn_s1, bcn_activities):
    # An overnight flight landing at 01:00 on day 2 leaves day 1 with nothing
    overnight = bcn_f1.model_copy(update={
        "outbound_arrive": datetime(2026, 10, 11, 1, 0, tzinfo=ZoneInfo("Europe/Madrid"))
    })
    result = build_schedule(barcelona_request, overnight, bcn_s1, bcn_activities, [])
    day1 = result.days[0].items
    assert len(day1) == 1
    assert (day1[0].kind, day1[0].title) == ("rest", "Free time")