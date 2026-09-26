"""Tests for the deterministic validator.

Pattern: start from a plan that passes every rule, break exactly one thing,
and check that exactly the expected violation appears.
"""

from datetime import date, datetime
from typing import get_args
from zoneinfo import ZoneInfo

from core.budget import compute_budget
from core.validator import VIOLATION_OWNER, check_currency, check_stay, validate_plan
from providers.mock import MockStayProvider
from schemas.models import DayPlan
from schemas.state import ViolationCode

START = date(2026, 10, 10)
END = date(2026, 10, 13)


def _codes(violations):
    return [v.code for v in violations]


# ---------- The most important test ----------
def test_good_plan_has_no_violations(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary
):
    budget = compute_budget(
        lisbon_request, lisbon_flight, lisbon_stay, good_lisbon_itinerary, lisbon_activities
    )
    violations = validate_plan(
        lisbon_request, lisbon_flight, lisbon_stay,
        good_lisbon_itinerary, lisbon_activities, budget,
    )
    assert violations == []


# ---------- Flight window ----------
def test_activity_before_landing(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary, make_item
):
    # Lands 09:00, so nothing (except the transfer) may start before 10:30
    good_lisbon_itinerary[0].items.append(
        make_item("activity", "Tram 28", "morning", "10:20", "10:50", "LIS-A11")
    )
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert _codes(v) == ["activity_before_landing"]


def test_activity_after_departure(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary, make_item
):
    # Return flight 20:30, so activities must end by 17:30
    good_lisbon_itinerary[3].items = [
        make_item("activity", "National Tile Museum", "morning", "10:00", "11:30", "LIS-A08"),
        make_item("meal", "Lunch", "afternoon", "12:00", "13:00"),
        make_item("activity", "Tram 28", "afternoon", "16:30", "17:45", "LIS-A11"),
        make_item("transfer", "Hotel to airport", "evening", "18:00", "19:00"),
    ]
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert _codes(v) == ["activity_after_departure"]


# ---------- Activities ----------
def test_activity_on_closed_monday(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary, make_item
):
    # 12 October 2026 is a Monday; Jeronimos Monastery is closed on Mondays
    good_lisbon_itinerary[2].items.append(
        make_item("activity", "Jeronimos Monastery", "afternoon", "15:00", "16:30", "LIS-A01")
    )
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert _codes(v) == ["activity_on_closed_day"]


def test_activity_outside_opening_hours(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary, make_item
):
    # Fado opens at 20:00
    good_lisbon_itinerary[1].items.append(
        make_item("activity", "Fado evening", "evening", "18:00", "19:30", "LIS-A06")
    )
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert _codes(v) == ["activity_outside_hours"]


def test_unknown_activity_id_is_caught(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary, make_item
):
    # Simulates an LLM inventing an activity that is not in the candidate pool
    good_lisbon_itinerary[2].items.append(
        make_item("activity", "Invented place", "afternoon", "16:00", "17:00", "LIS-A99")
    )
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert _codes(v) == ["unknown_activity"]


def test_overlapping_items(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary, make_item
):
    # Belem Tower runs 14:00-15:00 on day 2
    good_lisbon_itinerary[1].items.append(
        make_item("meal", "Coffee", "afternoon", "14:30", "15:15")
    )
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert _codes(v) == ["overlapping_items"]


# ---------- Days ----------
def test_missing_day_is_reported(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary
):
    del good_lisbon_itinerary[2]  # remove Monday
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert _codes(v) == ["empty_day"]


def test_day_outside_trip_is_reported(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary, make_item
):
    good_lisbon_itinerary.append(
        DayPlan(date=date(2026, 10, 14), items=[
            make_item("meal", "Breakfast", "morning", "09:00", "10:00")
        ])
    )
    v = validate_plan(lisbon_request, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, None)
    assert "activity_outside_trip" in _codes(v)


# ---------- Stay ----------
def test_late_arrival_with_no_late_checkin(barcelona_request, bcn_late_flight, bcn_hostel):
    assert _codes(check_stay(barcelona_request, bcn_late_flight, bcn_hostel)) == ["no_late_checkin"]


def test_late_arrival_is_fine_when_hotel_allows_it(barcelona_request, bcn_late_flight):
    stays = MockStayProvider().search_stays("Barcelona", START, END)
    gothic = next(s for s in stays if s.id == "BCN-S1")  # late check-in OK
    assert check_stay(barcelona_request, bcn_late_flight, gothic) == []


def test_after_midnight_arrival_counts_as_late(barcelona_request, bcn_late_flight, bcn_hostel):
    after_midnight = bcn_late_flight.model_copy(update={
        "outbound_arrive": datetime(2026, 10, 11, 0, 30, tzinfo=ZoneInfo("Europe/Madrid"))
    })
    assert _codes(check_stay(barcelona_request, after_midnight, bcn_hostel)) == ["no_late_checkin"]


def test_nights_without_a_hotel(lisbon_request, lisbon_flight):
    assert _codes(check_stay(lisbon_request, lisbon_flight, None)) == ["stay_dates_mismatch"]


# ---------- Budget and currency ----------
def test_over_budget(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary
):
    tight = lisbon_request.model_copy(update={"budget_amount": 1500})  # plan costs 1601.60
    budget = compute_budget(tight, lisbon_flight, lisbon_stay,
                            good_lisbon_itinerary, lisbon_activities)
    v = validate_plan(tight, lisbon_flight, lisbon_stay,
                      good_lisbon_itinerary, lisbon_activities, budget)
    assert _codes(v) == ["over_budget"]


def test_currency_mismatch(lisbon_request, lisbon_flight, lisbon_stay):
    euro_stay = lisbon_stay.model_copy(update={"currency": "EUR"})
    v = check_currency(lisbon_request, lisbon_flight, euro_stay, [])
    assert _codes(v) == ["currency_mismatch"]
    assert v[0].item_ids == ["LIS-S1"]


# ---------- Routing table ----------
def test_every_violation_code_has_an_owner():
    # If someone adds a new code but forgets the routing entry, this fails
    assert set(get_args(ViolationCode)) == set(VIOLATION_OWNER)