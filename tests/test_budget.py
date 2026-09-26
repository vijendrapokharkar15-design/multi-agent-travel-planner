"""Tests for the deterministic budget calculation."""

from datetime import date

import pytest

from core.budget import CurrencyMismatchError, compute_budget, rooms_needed
from schemas.models import DayPlan, TripRequest

START = date(2026, 10, 10)
END = date(2026, 10, 13)


def _request(**overrides):
    fields = dict(
        origin="London",
        destination="Lisbon",
        start_date=START,
        end_date=END,
        travellers=2,
        budget_amount=1200,
    )
    fields.update(overrides)
    return TripRequest(**fields)


# ---------- The hand-worked example ----------
def test_hand_worked_example(lisbon_flight, lisbon_stay):
    b = compute_budget(_request(), lisbon_flight, lisbon_stay, [], [])
    assert b.flights == 236  # 118 x 2
    assert b.stay == 435  # 145 x 3 nights x 1 room
    assert b.activities == 0
    assert b.food_estimate == 400  # 50 x 4 days x 2
    assert b.local_transport_estimate == 128  # (10 x 4 + 12 x 2) x 2
    assert b.contingency == pytest.approx(119.9)
    assert b.total == pytest.approx(1318.9)
    assert b.per_traveller == pytest.approx(659.45)
    assert b.over_by == pytest.approx(118.9)


# ---------- Rooms ----------
@pytest.mark.parametrize("travellers, rooms", [(1, 1), (2, 1), (3, 2), (5, 3)])
def test_rooms_needed(travellers, rooms):
    assert rooms_needed(travellers) == rooms


def test_five_travellers_pay_for_three_rooms(lisbon_flight, lisbon_stay):
    b = compute_budget(_request(travellers=5), lisbon_flight, lisbon_stay, [], [])
    assert b.stay == 1305  # 145 x 3 nights x 3 rooms


# ---------- Budget scope flags ----------
def test_budget_that_excludes_flights(lisbon_flight, lisbon_stay):
    r = _request(budget_amount=1000, budget_includes_flights=False)
    b = compute_budget(r, lisbon_flight, lisbon_stay, [], [])
    assert b.total == pytest.approx(1318.9)  # the user still sees the full cost
    assert b.over_by == pytest.approx(82.9)  # 1318.9 - 236 flights - 1000


def test_no_budget_means_never_over(lisbon_flight, lisbon_stay):
    b = compute_budget(_request(budget_amount=None), lisbon_flight, lisbon_stay, [], [])
    assert b.over_by == 0


# ---------- Activities ----------
def test_only_scheduled_activities_are_charged(
    lisbon_request, lisbon_flight, lisbon_stay, lisbon_activities, good_lisbon_itinerary
):
    b = compute_budget(
        lisbon_request, lisbon_flight, lisbon_stay, good_lisbon_itinerary, lisbon_activities
    )
    # 11 scheduled activities cost 128.50 per person; the 12th (A10) is not scheduled
    assert b.activities == pytest.approx(257)
    assert b.total == pytest.approx(1601.6)
    assert b.over_by == 0  # budget is 2000


def test_unknown_activity_raises(lisbon_flight, lisbon_stay, lisbon_activities, make_item):
    itinerary = [
        DayPlan(date=START, items=[
            make_item("activity", "Invented place", "morning", "11:00", "12:00", "LIS-A99")
        ])
    ]
    with pytest.raises(ValueError, match="unknown activity"):
        compute_budget(_request(), lisbon_flight, lisbon_stay, itinerary, lisbon_activities)


# ---------- Edge cases ----------
def test_same_day_trip_without_flight_or_stay():
    r = _request(end_date=START, travellers=1, budget_amount=None)
    b = compute_budget(r, None, None, [], [])
    assert b.food_estimate == 50  # 50 x 1 day x 1
    assert b.local_transport_estimate == 34  # 10 x 1 + 12 x 2
    assert b.total == pytest.approx(92.4)  # 84 + 10% contingency


def test_non_gbp_request_raises(lisbon_flight, lisbon_stay):
    with pytest.raises(CurrencyMismatchError, match="GBP only"):
        compute_budget(_request(currency="EUR"), lisbon_flight, lisbon_stay, [], [])