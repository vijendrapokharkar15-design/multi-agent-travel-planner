"""Workflow schemas: violations, revision requests, trace entries and the graph state.

Travel data models live in schemas/models.py.
"""

import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from schemas.models import (
    ActivityCandidate,
    BudgetBreakdown,
    DayPlan,
    FlightOption,
    StayOption,
    TripRequest,
)

MAX_REVISIONS = 3

# ---------- Fixed-choice types ----------
ViolationCode = Literal[
    "activity_outside_trip",
    "activity_before_landing",
    "activity_after_departure",
    "activity_on_closed_day",
    "activity_outside_hours",
    "overlapping_items",
    "empty_day",
    "stay_dates_mismatch",
    "no_late_checkin",
    "over_budget",
    "currency_mismatch",
]

NodeName = Literal["flight_agent", "activities_agent", "stay_agent", "itinerary_agent"]
BudgetAction = Literal["cheaper_stay", "cheaper_flight", "drop_paid_activities"]
AgentStatus = Literal["success", "partial", "failed"]
RunStatus = Literal["running", "valid", "invalid", "failed"]


# ---------- Workflow objects ----------
class Violation(BaseModel):
    """One broken rule found by the validator."""

    code: ViolationCode
    message: str
    item_ids: list[str] = Field(default_factory=list)


class RevisionRequest(BaseModel):
    """What the router sends back to a node when validation fails."""

    target: NodeName
    codes: list[ViolationCode]
    reason: str
    budget_action: BudgetAction | None = None


class TraceEntry(BaseModel):
    """One record per node run. This is our 'result envelope'."""

    agent: str
    status: AgentStatus
    started_at: datetime
    latency_ms: int = Field(ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    summary: str
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    revision: int = Field(default=0, ge=0)  # which revision loop this ran in


# ---------- Graph state ----------
class TripState(TypedDict, total=False):
    # Input
    raw_request: str
    request: TripRequest

    # Agent outputs
    flight_options: list[FlightOption]
    selected_flight_id: str | None
    activity_candidates: list[ActivityCandidate]
    stay_options: list[StayOption]
    selected_stay_id: str | None
    itinerary: list[DayPlan]
    budget: BudgetBreakdown | None

    # Validation and revision (single writer each, so plain replacement)
    violations: list[Violation]
    revision_request: RevisionRequest | None
    revision_count: int
    status: RunStatus

    # Append-only keys (reducer = list concatenation)
    excluded_ids: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEntry], operator.add]
    warnings: Annotated[list[str], operator.add]