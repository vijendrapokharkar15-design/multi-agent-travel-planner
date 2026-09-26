"""Provider interfaces.

Agents depend on these abstract classes, never on a specific data source.
Mock and real providers both implement them, so they can be swapped
without changing any agent code.
"""

from abc import ABC, abstractmethod
from datetime import date

from schemas.models import ActivityCandidate, FlightOption, StayOption


class UnsupportedDestinationError(Exception):
    """Raised when a provider has no coverage for the requested route or city.

    This is different from an empty result: an empty list means the provider
    searched and found nothing, while this error means it cannot search at all.
    """


class FlightProvider(ABC):
    @abstractmethod
    def search_flights(
        self,
        origin: str,
        destination: str,
        depart_date: date,
        return_date: date,
    ) -> list[FlightOption]:
        """Return round-trip flight options for the given route and dates."""


class StayProvider(ABC):
    @abstractmethod
    def search_stays(
        self,
        destination: str,
        check_in: date,
        check_out: date,
    ) -> list[StayOption]:
        """Return accommodation options available for the whole stay."""


class ActivityProvider(ABC):
    @abstractmethod
    def search_activities(self, destination: str) -> list[ActivityCandidate]:
        """Return candidate activities in the destination."""