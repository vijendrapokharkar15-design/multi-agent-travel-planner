"""The evaluation set: 8 trips, each chosen to test something specific.

Dates are in November 2026, chosen so that weekdays matter:
  Mon 9, Tue 10, Fri 13, Sat 14, Tue 17, Fri 20, Tue 24 November.
Both systems get the same requests and the same mock data.
"""

from dataclasses import dataclass
from datetime import date

from schemas.models import TripRequest

# Fixed "today" so free-text dates like "9 November" always resolve the same way
EVAL_TODAY = date(2026, 10, 6)


@dataclass(frozen=True)
class EvalCase:
    id: str
    title: str
    tests: str  # what this case is designed to test
    request: TripRequest  # structured request: used by the baseline, and by the
                          # multi-agent system unless raw_text is set
    raw_text: str | None = None  # if set, the multi-agent system receives this text instead
    expect_unaffordable: bool = False  # the budget cannot be met; success = failing honestly


def _req(destination, start, end, travellers, budget, interests, style="mid-range", pace="balanced"):
    return TripRequest(
        origin="London", destination=destination, start_date=start, end_date=end,
        travellers=travellers, budget_amount=budget, interests=interests,
        style=style, pace=pace,
    )


CASES: list[EvalCase] = [
    EvalCase(
        id="01_standard_lisbon",
        title="Lisbon, 4 days, 2 people, £1,500",
        tests="The standard trip",
        request=_req("Lisbon", date(2026, 11, 10), date(2026, 11, 13), 2, 1500,
                     ["food", "architecture"]),
    ),
    EvalCase(
        id="02_tight_barcelona",
        title="Barcelona, 4 days, 2 people, £1,200",
        tests="Tight budget, so the revision loop should fire",
        request=_req("Barcelona", date(2026, 11, 13), date(2026, 11, 16), 2, 1200,
                     ["food", "nightlife"]),
    ),
    EvalCase(
        id="03_impossible_budget",
        title="Lisbon, 3 days, 2 people, £400",
        tests="Impossible budget: must say so honestly, without breaking anything",
        request=_req("Lisbon", date(2026, 11, 17), date(2026, 11, 19), 2, 400, ["history"]),
        expect_unaffordable=True,
    ),
    EvalCase(
        id="04_same_day",
        title="Barcelona, same-day trip, 1 person, £300",
        tests="No hotel needed; only some flights are possible",
        request=_req("Barcelona", date(2026, 11, 14), date(2026, 11, 14), 1, 300,
                     ["architecture"]),
    ),
    EvalCase(
        id="05_budget_style",
        title="Barcelona, 3 days, 2 people, budget style, £700",
        # £700 makes Demo Air + the hostel affordable (about £682 with contingency),
        # while the cheapest flight lands at 23:35, where the hostel has no late check-in.
        # (The first version used £600, which turned out to be impossible for any plan.)
        tests="Tempts the cheapest late flight: the hotel must allow late check-in",
        request=_req("Barcelona", date(2026, 11, 20), date(2026, 11, 22), 2, 700,
                     ["nightlife", "food"], style="budget"),
    ),
    EvalCase(
        id="06_monday_closures",
        title="Lisbon, Monday to Wednesday, history and art",
        tests="Three Lisbon sights are closed on Mondays",
        request=_req("Lisbon", date(2026, 11, 9), date(2026, 11, 11), 2, 1500,
                     ["history", "art"]),
    ),
    EvalCase(
        id="07_group_of_five",
        title="Barcelona, 4 days, 5 people, £4,000",
        tests="Rooms (3 for 5 people) and per-person costs at a larger group size",
        request=_req("Barcelona", date(2026, 11, 24), date(2026, 11, 27), 5, 4000,
                     ["architecture", "food"]),
    ),
    EvalCase(
        id="08_free_text",
        title="Free text: 3 days in Lisbon for 2, about £1,200",
        tests="The parser path: the request must be understood correctly",
        request=_req("Lisbon", date(2026, 11, 9), date(2026, 11, 11), 2, 1200,
                     ["food", "views"]),
        raw_text="3 days in Lisbon from 9 November for 2 of us, around £1,200, "
                 "we love food and views",
    ),
]