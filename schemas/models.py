"""Pydantic data models for the travel planner.

These describe travel data: requests, options, plans and costs.
Workflow objects (graph state, violations, revisions) live in schemas/state.py.
"""

from datetime import date, datetime, time
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

MAX_TRIP_DAYS = 7

# ---------- Shared fixed-choice types ----------
Provenance = Literal["live_api", "mock", "estimate"]
Pace = Literal["relaxed", "balanced", "packed"]
Style = Literal["budget", "mid-range", "luxury"]
FlightLabel = Literal["cheapest", "fastest", "best_value"]
StayLabel = Literal["cheapest", "best_location", "best_value"]
Slot = Literal["morning", "afternoon", "evening"]
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
ItemKind = Literal["activity", "meal", "transfer", "rest"]

# Index matches date.weekday(), where Monday is 0
WEEKDAYS: tuple[Weekday, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


# ---------- Reusable validation helpers ----------
def _currency_code(value: str) -> str:
    value = value.strip().upper()
    if len(value) != 3 or not value.isalpha():
        raise ValueError("currency must be a 3-letter code like GBP")
    return value


CurrencyCode = Annotated[str, AfterValidator(_currency_code)]


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a time zone")
    return value


# ---------- Request ----------
class TripRequest(BaseModel):
    origin: str = Field(min_length=2)
    destination: str = Field(min_length=2)
    start_date: date
    end_date: date
    travellers: int = Field(default=1, ge=1)
    budget_amount: float | None = Field(default=None, gt=0)
    currency: CurrencyCode = "GBP"
    budget_includes_flights: bool = True
    budget_includes_stay: bool = True
    style: Style = "mid-range"
    interests: list[str] = Field(default_factory=list)
    pace: Pace = "balanced"
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_dates(self) -> "TripRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date")
        if self.num_days > MAX_TRIP_DAYS:
            raise ValueError(f"trips are limited to {MAX_TRIP_DAYS} days")
        return self

    @property
    def num_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def num_nights(self) -> int:
        return (self.end_date - self.start_date).days


# ---------- Options returned by providers ----------
class FlightOption(BaseModel):
    id: str
    airline: str
    outbound_flight_number: str
    outbound_depart: datetime
    outbound_arrive: datetime
    outbound_stops: int = Field(default=0, ge=0)
    return_flight_number: str
    return_depart: datetime
    return_arrive: datetime
    return_stops: int = Field(default=0, ge=0)
    price_per_person: float = Field(gt=0)
    currency: CurrencyCode
    provenance: Provenance
    label: FlightLabel | None = None  # set by the Flight Agent
    reason: str | None = None  # one-line explanation from the LLM

    @field_validator("outbound_depart", "outbound_arrive", "return_depart", "return_arrive")
    @classmethod
    def must_have_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def check_times(self) -> "FlightOption":
        if self.outbound_arrive <= self.outbound_depart:
            raise ValueError("outbound flight must arrive after it departs")
        if self.return_arrive <= self.return_depart:
            raise ValueError("return flight must arrive after it departs")
        if self.return_depart <= self.outbound_arrive:
            raise ValueError("return flight must depart after the outbound arrives")
        return self

    @property
    def outbound_duration_mins(self) -> int:
        return int((self.outbound_arrive - self.outbound_depart).total_seconds() // 60)

    @property
    def return_duration_mins(self) -> int:
        return int((self.return_arrive - self.return_depart).total_seconds() // 60)


class StayOption(BaseModel):
    id: str
    name: str
    area: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    price_per_night: float = Field(gt=0)
    currency: CurrencyCode
    rating: float | None = Field(default=None, ge=0)
    late_checkin_ok: bool = False
    provenance: Provenance
    label: StayLabel | None = None  # set by the Stay Agent
    reason: str | None = None

    def total_price(self, nights: int) -> float:
        return round(self.price_per_night * nights, 2)


class ActivityCandidate(BaseModel):
    id: str
    name: str
    area: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    category: str
    interest_tags: list[str] = Field(default_factory=list)
    typical_duration_mins: int = Field(gt=0)
    price_per_person: float = Field(ge=0)
    currency: CurrencyCode
    closed_days: list[Weekday] = Field(default_factory=list)
    open_time: time | None = None
    close_time: time | None = None
    booking_needed: bool = False
    provenance: Provenance

    @model_validator(mode="after")
    def check_hours(self) -> "ActivityCandidate":
        if self.open_time and self.close_time and self.close_time <= self.open_time:
            raise ValueError("close_time must be after open_time")
        return self


# ---------- Plan ----------
class ItineraryItem(BaseModel):
    kind: ItemKind
    title: str
    slot: Slot
    activity_id: str | None = None  # required when kind is "activity"
    start_time: time | None = None  # filled in by code, not the LLM
    end_time: time | None = None

    @model_validator(mode="after")
    def check_item(self) -> "ItineraryItem":
        if self.kind == "activity" and not self.activity_id:
            raise ValueError("activity items need an activity_id")
        if self.start_time and self.end_time and self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


class DayPlan(BaseModel):
    date: date
    items: list[ItineraryItem] = Field(default_factory=list)
    notes: str | None = None


class BudgetBreakdown(BaseModel):
    currency: CurrencyCode
    flights: float = Field(ge=0)
    stay: float = Field(ge=0)
    activities: float = Field(ge=0)
    food_estimate: float = Field(ge=0)
    local_transport_estimate: float = Field(ge=0)
    contingency: float = Field(ge=0)
    total: float = Field(ge=0)
    per_traveller: float = Field(ge=0)
    budget_amount: float | None = None
    over_by: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_total(self) -> "BudgetBreakdown":
        parts = (
            self.flights
            + self.stay
            + self.activities
            + self.food_estimate
            + self.local_transport_estimate
            + self.contingency
        )
        if abs(parts - self.total) > 0.01:
            raise ValueError("total must equal the sum of the categories")
        return self