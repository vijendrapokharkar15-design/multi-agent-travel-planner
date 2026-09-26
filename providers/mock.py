"""Mock providers backed by the JSON files in data/mock/.

Deterministic, offline and clearly labelled as mock data.
Demo coverage: flights from London to Lisbon or Barcelona.
"""

import json
import re
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from providers.base import (
    ActivityProvider,
    FlightProvider,
    StayProvider,
    UnsupportedDestinationError,
)
from schemas.models import ActivityCandidate, FlightOption, StayOption

MOCK_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"
SUPPORTED_ORIGIN = "london"


# ---------- Helpers ----------
def _city_key(name: str) -> str:
    """Turn 'Lisbon, Portugal' into 'lisbon'. Rejects anything that isn't a plain name.

    The check matters because this value can come from user input or an LLM,
    and it is used to build a file path.
    """
    key = name.split(",")[0].strip().lower()
    if not re.fullmatch(r"[a-z][a-z \-]*", key):
        raise UnsupportedDestinationError(f"Invalid place name: {name!r}")
    return key


@lru_cache(maxsize=None)
def load_city(city_key: str) -> dict:
    """Load one city's mock data once and cache it. Callers must not modify it."""
    path = MOCK_DIR / f"{city_key}.json"
    if not path.exists():
        supported = ", ".join(sorted(p.stem.title() for p in MOCK_DIR.glob("*.json")))
        raise UnsupportedDestinationError(
            f"Demo data covers {supported} only, not '{city_key.title()}'."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _with_labels(item: dict, city: dict) -> dict:
    """Copy an item and add the fields every mock record must carry."""
    return {**item, "provenance": "mock", "currency": city["currency"]}


def _local(day: date, hhmm: str, tz: ZoneInfo) -> datetime:
    """Build a timezone-aware datetime from a date, 'HH:MM' and a zone."""
    return datetime.combine(day, time.fromisoformat(hhmm), tzinfo=tz)


def _arrival(depart: datetime, duration_mins: int, arrival_tz: ZoneInfo) -> datetime:
    """Add a flight duration safely, then express the result on the arrival clock.

    We add the duration in UTC because adding a timedelta directly to a
    ZoneInfo datetime adds 'wall clock' time, which is wrong across a
    daylight-saving change.
    """
    arrive_utc = depart.astimezone(timezone.utc) + timedelta(minutes=duration_mins)
    return arrive_utc.astimezone(arrival_tz)


# ---------- Providers ----------
class MockFlightProvider(FlightProvider):
    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: date,
        return_date: date,
    ) -> list[FlightOption]:
        if _city_key(origin) != SUPPORTED_ORIGIN:
            raise UnsupportedDestinationError("Demo flights depart from London only.")

        city = load_city(_city_key(destination))
        origin_tz = ZoneInfo(city["origin_timezone"])
        dest_tz = ZoneInfo(city["timezone"])

        options: list[FlightOption] = []
        for f in city["flights"]:
            out_depart = _local(depart_date, f["outbound_depart_local"], origin_tz)
            ret_depart = _local(return_date, f["return_depart_local"], dest_tz)
            try:
                options.append(
                    FlightOption(
                        id=f["id"],
                        airline=f["airline"],
                        outbound_flight_number=f["outbound_flight_number"],
                        outbound_depart=out_depart,
                        outbound_arrive=_arrival(out_depart, f["outbound_duration_mins"], dest_tz),
                        outbound_stops=f["outbound_stops"],
                        return_flight_number=f["return_flight_number"],
                        return_depart=ret_depart,
                        return_arrive=_arrival(ret_depart, f["return_duration_mins"], origin_tz),
                        return_stops=f["return_stops"],
                        price_per_person=f["price_per_person"],
                        currency=city["currency"],
                        provenance="mock",
                    )
                )
            except ValidationError:
                # Impossible for these dates (e.g. a same-day trip where the
                # return leaves before the outbound lands). Skip, don't crash.
                continue
        return options


class MockStayProvider(StayProvider):
    def search_stays(
        self,
        destination: str,
        check_in: date,
        check_out: date,
    ) -> list[StayOption]:
        if (check_out - check_in).days <= 0:
            return []  # same-day trip: no hotel needed
        city = load_city(_city_key(destination))
        # Mock assumption: every property is available for any dates.
        return [StayOption(**_with_labels(s, city)) for s in city["stays"]]


class MockActivityProvider(ActivityProvider):
    def search_activities(self, destination: str) -> list[ActivityCandidate]:
        city = load_city(_city_key(destination))
        return [ActivityCandidate(**_with_labels(a, city)) for a in city["activities"]]