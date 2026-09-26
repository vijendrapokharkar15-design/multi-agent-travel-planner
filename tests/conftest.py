"""Shared pytest fixtures. pytest loads this file automatically,
so any test can use these just by naming them as parameters."""

from datetime import date

import pytest

from providers.mock import MockActivityProvider, MockFlightProvider, MockStayProvider
from schemas.models import DayPlan, ItineraryItem, TripRequest

START = date(2026, 10, 10)  # a Saturday
END = date(2026, 10, 13)  # a Tuesday, so the trip includes Monday 12 October


# ---------- Small helpers ----------
def _pick(options, option_id):
    return next(o for o in options if o.id == option_id)


def _item(kind, title, slot, start, end, activity_id=None):
    return ItineraryItem(
        kind=kind,
        title=title,
        slot=slot,
        activity_id=activity_id,
        start_time=start,
        end_time=end,
    )


@pytest.fixture
def make_item():
    """A 'factory fixture': tests call it to build itinerary items."""
    return _item


# ---------- Requests ----------
@pytest.fixture
def lisbon_request():
    return TripRequest(
        origin="London",
        destination="Lisbon",
        start_date=START,
        end_date=END,
        travellers=2,
        budget_amount=2000,
    )


@pytest.fixture
def barcelona_request():
    return TripRequest(
        origin="London",
        destination="Barcelona",
        start_date=START,
        end_date=END,
        travellers=2,
        budget_amount=1200,
    )


# ---------- Lisbon data ----------
@pytest.fixture
def lisbon_flight(lisbon_request):
    flights = MockFlightProvider().search_flights("London", "Lisbon", START, END)
    return _pick(flights, "LIS-F1")  # lands 09:00, returns 20:30


@pytest.fixture
def lisbon_stay():
    stays = MockStayProvider().search_stays("Lisbon", START, END)
    return _pick(stays, "LIS-S1")  # Baixa, late check-in OK


@pytest.fixture
def lisbon_activities():
    return MockActivityProvider().search_activities("Lisbon")


# ---------- Barcelona data ----------
@pytest.fixture
def bcn_late_flight():
    flights = MockFlightProvider().search_flights("London", "Barcelona", START, END)
    return _pick(flights, "BCN-F3")  # lands 23:35 local


@pytest.fixture
def bcn_hostel():
    stays = MockStayProvider().search_stays("Barcelona", START, END)
    return _pick(stays, "BCN-S2")  # no late check-in


@pytest.fixture
def bcn_activities():
    return MockActivityProvider().search_activities("Barcelona")


# ---------- A hand-built itinerary that should pass every check ----------
@pytest.fixture
def good_lisbon_itinerary():
    """Designed around LIS-F1: land 09:00 (activities from 10:30),
    return flight 20:30 (activities must end by 17:30).
    Monday avoids the three places closed on Mondays."""
    return [
        DayPlan(date=date(2026, 10, 10), items=[
            _item("transfer", "Airport to hotel", "morning", "09:15", "10:15"),
            _item("activity", "Sao Jorge Castle", "morning", "11:00", "12:30", "LIS-A05"),
            _item("meal", "Lunch", "afternoon", "12:45", "13:45"),
            _item("activity", "Santa Luzia viewpoint", "afternoon", "14:00", "14:30", "LIS-A07"),
            _item("activity", "Fado evening", "evening", "20:00", "22:00", "LIS-A06"),
        ]),
        DayPlan(date=date(2026, 10, 11), items=[
            _item("activity", "Jeronimos Monastery", "morning", "10:00", "11:30", "LIS-A01"),
            _item("activity", "Custard tart tasting", "morning", "11:45", "12:30", "LIS-A03"),
            _item("meal", "Lunch", "afternoon", "12:45", "13:45"),
            _item("activity", "Belem Tower", "afternoon", "14:00", "15:00", "LIS-A02"),
            _item("activity", "Riverside walk", "afternoon", "15:30", "16:30", "LIS-A04"),
        ]),
        DayPlan(date=date(2026, 10, 12), items=[
            _item("activity", "Tram 28", "morning", "10:00", "11:00", "LIS-A11"),
            _item("activity", "Gulbenkian Museum", "morning", "11:30", "13:30", "LIS-A12"),
            _item("meal", "Lunch", "afternoon", "13:45", "14:45"),
            _item("activity", "Food hall dinner", "evening", "19:00", "20:30", "LIS-A09"),
        ]),
        DayPlan(date=date(2026, 10, 13), items=[
            _item("activity", "National Tile Museum", "morning", "10:00", "11:30", "LIS-A08"),
            _item("meal", "Lunch", "afternoon", "12:00", "13:00"),
            _item("transfer", "Hotel to airport", "afternoon", "17:30", "18:30"),
        ]),
    ]