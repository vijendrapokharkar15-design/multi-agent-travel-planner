"""Tests for the mock providers: data loading, labelling and time zones."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from providers.base import UnsupportedDestinationError
from providers.mock import (
    MockActivityProvider,
    MockFlightProvider,
    MockStayProvider,
    _arrival,
)

START = date(2026, 10, 10)
END = date(2026, 10, 13)
LONDON = ZoneInfo("Europe/London")
MADRID = ZoneInfo("Europe/Madrid")


# ---------- Time zones ----------
def test_barcelona_late_flight_lands_2335_local(bcn_late_flight):
    arrive = bcn_late_flight.outbound_arrive
    assert (arrive.hour, arrive.minute) == (23, 35)
    assert arrive.utcoffset() == timedelta(hours=2)  # Barcelona summer time
    assert bcn_late_flight.outbound_duration_mins == 125


def test_lisbon_uses_the_same_clock_as_london(lisbon_flight):
    # LIS-F1: departs 06:15, flies 165 minutes, lands 09:00 on the same clock
    assert lisbon_flight.outbound_depart.utcoffset() == timedelta(hours=1)
    assert lisbon_flight.outbound_arrive.utcoffset() == timedelta(hours=1)
    assert (lisbon_flight.outbound_arrive.hour, lisbon_flight.outbound_arrive.minute) == (9, 0)


def test_return_flight_arrives_on_london_clock():
    flights = MockFlightProvider().search_flights("London", "Barcelona", START, END)
    bcn_f1 = next(f for f in flights if f.id == "BCN-F1")
    # Departs Barcelona 19:45 local, flies 130 minutes, lands London 20:55 local
    assert (bcn_f1.return_arrive.hour, bcn_f1.return_arrive.minute) == (20, 55)
    assert bcn_f1.return_arrive.utcoffset() == timedelta(hours=1)


def test_arrival_is_correct_across_daylight_saving_change():
    # Clocks go back on Sunday 25 October 2026. Leaving London at 00:30
    # and flying 125 minutes lands at 02:35 Barcelona time. Naive clock-face
    # addition would give 03:35.
    depart = datetime(2026, 10, 25, 0, 30, tzinfo=LONDON)
    arrive = _arrival(depart, 125, MADRID)
    assert (arrive.hour, arrive.minute) == (2, 35)
    assert arrive.utcoffset() == timedelta(hours=1)  # Barcelona now on winter time


# ---------- Data loading and labelling ----------
@pytest.mark.parametrize("city", ["Lisbon", "Barcelona"])
def test_each_city_has_expected_counts(city):
    assert len(MockFlightProvider().search_flights("London", city, START, END)) == 4
    assert len(MockStayProvider().search_stays(city, START, END)) == 4
    assert len(MockActivityProvider().search_activities(city)) == 12


@pytest.mark.parametrize("city", ["Lisbon", "Barcelona"])
def test_everything_is_labelled_mock_and_gbp(city):
    items = [
        *MockFlightProvider().search_flights("London", city, START, END),
        *MockStayProvider().search_stays(city, START, END),
        *MockActivityProvider().search_activities(city),
    ]
    assert all(i.provenance == "mock" for i in items)
    assert all(i.currency == "GBP" for i in items)


def test_destination_name_is_forgiving():
    activities = MockActivityProvider().search_activities("  lisbon, Portugal ")
    assert len(activities) == 12


def test_stay_total_price_for_three_nights(lisbon_stay):
    assert lisbon_stay.total_price(3) == 435  # 145 x 3


# ---------- Unsupported input ----------
def test_unsupported_destination_raises():
    with pytest.raises(UnsupportedDestinationError, match="Demo data covers"):
        MockActivityProvider().search_activities("Paris")


def test_non_london_origin_raises():
    with pytest.raises(UnsupportedDestinationError, match="London only"):
        MockFlightProvider().search_flights("Manchester", "Lisbon", START, END)


def test_path_like_destination_is_rejected():
    with pytest.raises(UnsupportedDestinationError, match="Invalid place name"):
        MockActivityProvider().search_activities("../../secrets")


# ---------- Same-day trips ----------
def test_same_day_trip_needs_no_hotel():
    assert MockStayProvider().search_stays("Lisbon", START, START) == []


def test_same_day_trip_skips_impossible_flights():
    flights = MockFlightProvider().search_flights("London", "Lisbon", START, START)
    # F3 lands 22:35 but returns 06:05; F4 lands 14:30 but returns 13:00.
    assert {f.id for f in flights} == {"LIS-F1", "LIS-F2"}