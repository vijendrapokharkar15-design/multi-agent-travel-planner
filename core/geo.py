"""Distance and travel-time estimates.

Straight-line (haversine) distance between coordinates, plus a simple,
documented rule for turning distance into travel time. These are estimates,
not routing: good enough for choosing a hotel and spacing out a day.
"""

import math
from typing import Protocol

EARTH_RADIUS_KM = 6371.0

# Travel-time rule of thumb (documented estimate, not live transport data)
WALK_MAX_KM = 1.5  # up to this, assume walking
WALK_KMH = 4.5
TRANSIT_KMH = 15.0  # average door-to-door speed on city public transport
TRANSIT_OVERHEAD_MINS = 10  # walking to the stop plus waiting


class HasLocation(Protocol):
    lat: float
    lon: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_km(a: HasLocation, b: HasLocation) -> float:
    return haversine_km(a.lat, a.lon, b.lat, b.lon)


def mean_distance_km(origin: HasLocation, places: list[HasLocation]) -> float | None:
    """Average distance from one point to several others. None if there are none."""
    if not places:
        return None
    return sum(distance_km(origin, p) for p in places) / len(places)


def travel_minutes(km: float) -> int:
    """Estimated minutes to get between two points in a city."""
    if km <= WALK_MAX_KM:
        return math.ceil(km / WALK_KMH * 60)
    return math.ceil(km / TRANSIT_KMH * 60 + TRANSIT_OVERHEAD_MINS)