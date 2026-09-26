"""Tests for the rules built into the Pydantic schemas."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from schemas.models import BudgetBreakdown, FlightOption, ItineraryItem, TripRequest
from schemas.state import Violation

LONDON = ZoneInfo("Europe/London")
LISBON = ZoneInfo("Europe/Lisbon")


def _request(**overrides):
    fields = dict(
        origin="London",
        destination="Lisbon",
        start_date=date(2026, 10, 10),
        end_date=date(2026, 10, 13),
    )
    fields.update(overrides)
    return TripRequest(**fields)


def _flight(**overrides):
    fields = dict(
        id="T-F1",
        airline="Test Air",
        outbound_flight_number="TA1",
        outbound_depart=datetime(2026, 10, 10, 7, 0, tzinfo=LONDON),
        outbound_arrive=datetime(2026, 10, 10, 9, 45, tzinfo=LISBON),
        return_flight_number="TA2",
        return_depart=datetime(2026, 10, 13, 18, 0, tzinfo=LISBON),
        return_arrive=datetime(2026, 10, 13, 20, 40, tzinfo=LONDON),
        price_per_person=100,
        currency="GBP",
        provenance="mock",
    )
    fields.update(overrides)
    return FlightOption(**fields)


# ---------- TripRequest ----------
def test_request_defaults_and_counts():
    r = _request()
    assert r.currency == "GBP"
    assert r.travellers == 1
    assert r.style == "mid-range"
    assert r.pace == "balanced"
    assert (r.num_days, r.num_nights) == (4, 3)


def test_seven_day_trip_allowed():
    assert _request(end_date=date(2026, 10, 16)).num_days == 7


def test_eight_day_trip_rejected():
    with pytest.raises(ValidationError, match="limited to 7 days"):
        _request(end_date=date(2026, 10, 17))


def test_end_before_start_rejected():
    with pytest.raises(ValidationError, match="cannot be before"):
        _request(end_date=date(2026, 10, 9))


def test_same_day_trip_has_no_nights():
    r = _request(end_date=date(2026, 10, 10))
    assert (r.num_days, r.num_nights) == (1, 0)


def test_currency_is_normalised():
    assert _request(currency=" gbp ").currency == "GBP"


def test_invalid_currency_rejected():
    with pytest.raises(ValidationError, match="3-letter code"):
        _request(currency="pounds")


def test_zero_travellers_rejected():
    with pytest.raises(ValidationError):
        _request(travellers=0)


# ---------- FlightOption ----------
def test_valid_flight_duration_across_time_zones():
    # Lisbon uses the same clock as London, so 07:00 -> 09:45 is 165 minutes
    assert _flight().outbound_duration_mins == 165


def test_flight_without_time_zone_rejected():
    with pytest.raises(ValidationError, match="time zone"):
        _flight(outbound_depart=datetime(2026, 10, 10, 7, 0))


def test_flight_arriving_before_departure_rejected():
    with pytest.raises(ValidationError, match="must arrive after"):
        _flight(outbound_arrive=datetime(2026, 10, 10, 6, 0, tzinfo=LISBON))


def test_return_before_outbound_lands_rejected():
    with pytest.raises(ValidationError, match="after the outbound arrives"):
        _flight(
            return_depart=datetime(2026, 10, 10, 9, 0, tzinfo=LISBON),
            return_arrive=datetime(2026, 10, 10, 11, 40, tzinfo=LONDON),
        )


# ---------- ItineraryItem ----------
def test_activity_item_needs_activity_id():
    with pytest.raises(ValidationError, match="need an activity_id"):
        ItineraryItem(kind="activity", title="Museum", slot="morning")


def test_item_end_before_start_rejected():
    with pytest.raises(ValidationError, match="end_time must be after"):
        ItineraryItem(kind="meal", title="Lunch", slot="afternoon",
                      start_time="13:00", end_time="12:00")


def test_invalid_slot_rejected():
    with pytest.raises(ValidationError):
        ItineraryItem(kind="meal", title="Lunch", slot="midday")


# ---------- BudgetBreakdown ----------
def test_budget_total_must_match_categories():
    with pytest.raises(ValidationError, match="sum of the categories"):
        BudgetBreakdown(
            currency="GBP", flights=100, stay=100, activities=0, food_estimate=0,
            local_transport_estimate=0, contingency=0, total=999, per_traveller=999,
        )


# ---------- Violation ----------
def test_unknown_violation_code_rejected():
    with pytest.raises(ValidationError):
        Violation(code="too_expensive", message="made-up code")