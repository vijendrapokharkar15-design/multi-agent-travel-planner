"""Activities Agent.

Builds a ranked CANDIDATE POOL for the traveller. It does not schedule;
the Itinerary Agent does that once flight times and the hotel are known.
"""

import math
from datetime import timedelta

from pydantic import BaseModel

from agents.common import DATA_NOTICE, Timer, data_block, trace_entry
from agents.llm import LLM, LLMError
from providers.base import ActivityProvider, UnsupportedDestinationError
from schemas.models import WEEKDAYS, ActivityCandidate, TripRequest
from schemas.state import TripState

AGENT = "activities_agent"
PER_DAY = {"relaxed": 2, "balanced": 3, "packed": 4}
SPARE_RATIO = 0.5  # extra candidates so the scheduler can work around closures
MIN_FREE = 2  # keep free options available for the "drop paid activities" fix


# ---------- What the LLM is allowed to return ----------
class ActivityPick(BaseModel):
    id: str
    reason: str


class ActivityDecision(BaseModel):
    picks: list[ActivityPick]
    note: str


SYSTEM_PROMPT = f"""You are the Activities Agent in a travel planner.
Select and rank activities from the candidates provided that fit this traveller.
- Match the traveller's interests, including loose matches
  (for example 'wine' fits food and tapas, 'photography' fits viewpoints).
- Keep the selection varied; avoid many activities of the same kind.
- For a budget traveller, prefer free and cheap options.
- If no interests are given, choose a balanced mix for a first-time visitor.
- picks: best fit first, each with one short sentence on why it suits this traveller.
- note: one sentence on how well the interests could be matched.
  Say plainly if an interest has no good match in this city.
You are not scheduling: do not mention days or times.
Only use IDs that appear in the candidates. Never invent activities or prices.
{DATA_NOTICE}"""


# ---------- Deterministic helpers ----------
def _trip_weekdays(request: TripRequest) -> set[str]:
    return {
        WEEKDAYS[(request.start_date + timedelta(days=i)).weekday()]
        for i in range(request.num_days)
    }


def _open_on_some_trip_day(activity: ActivityCandidate, weekdays: set[str]) -> bool:
    return not weekdays.issubset(activity.closed_days)


def _target_size(request: TripRequest, available: int) -> int:
    wanted = math.ceil(PER_DAY[request.pace] * request.num_days * (1 + SPARE_RATIO))
    return min(wanted, available)


def _matched_interests(activity: ActivityCandidate, interests: list[str]) -> list[str]:
    tags = {t.lower() for t in activity.interest_tags} | {activity.category.lower()}
    return [i for i in interests if i.lower() in tags]


def _rank_by_score(options: list[ActivityCandidate], interests: list[str]) -> list[ActivityCandidate]:
    """Fallback ranking: most matched interests first, then cheapest."""
    return sorted(
        options,
        key=lambda a: (-len(_matched_interests(a, interests)), a.price_per_person, a.id),
    )


def _describe(a: ActivityCandidate) -> str:
    booking = " | booking needed" if a.booking_needed else ""
    return (
        f"{a.id} | {a.name} | area: {a.area} | {a.category} | tags: {', '.join(a.interest_tags)} | "
        f"{a.typical_duration_mins} mins | {a.currency} {a.price_per_person:.2f} per person{booking}"
    )


def _failed(timer: Timer, message: str, revision: int) -> dict:
    return {
        "activity_candidates": [],
        "trace": [trace_entry(AGENT, "failed", timer, message, revision=revision)],
        "warnings": [message],
    }


# ---------- The node ----------
def run_activities_agent(state: TripState, provider: ActivityProvider, llm: LLM) -> dict:
    timer = Timer()
    request = state["request"]
    revision = state.get("revision_count", 0)
    interests = request.interests

    # 1. Fetch
    try:
        options = provider.search_activities(request.destination)
    except UnsupportedDestinationError as error:
        return _failed(timer, str(error), revision)

    # 2. Pre-filter facts
    excluded = set(state.get("excluded_ids", []))
    weekdays = _trip_weekdays(request)
    options = [
        a for a in options
        if a.id not in excluded and _open_on_some_trip_day(a, weekdays)
    ]
    if not options:
        return _failed(timer, "No activities are open during these dates.", revision)

    # 3. Pool size
    target = _target_size(request, len(options))
    by_id = {a.id: a for a in options}

    # 4. LLM picks and ranks
    user_prompt = (
        f"Traveller: interests={interests or 'none given'}, style={request.style}, "
        f"pace={request.pace}, travellers={request.travellers}.\n"
        f"Choose up to {target} activities.\n"
        "Candidates:\n" + data_block("\n".join(_describe(a) for a in options))
    )

    status, warnings, llm_result, note = "success", [], None, ""
    chosen: list[str] = []
    reasons: dict[str, str] = {}
    try:
        llm_result = llm.structured(SYSTEM_PROMPT, user_prompt, ActivityDecision)
        decision = llm_result.parsed
        note = decision.note
        unknown = 0
        for pick in decision.picks:
            if pick.id not in by_id:
                unknown += 1
            elif pick.id not in reasons:
                chosen.append(pick.id)
                reasons[pick.id] = pick.reason
        if unknown:
            warnings.append(f"Ignored {unknown} activity ID(s) not in the candidates.")
        chosen = chosen[:target]
    except LLMError as error:
        status = "partial"
        warnings.append(f"Used fallback ranking by interest match: {error}")

    # 5a. Top up by interest score if the pool is short
    ranked = _rank_by_score(options, interests)
    code_reasons: dict[str, str] = {}
    topped_up = 0
    for a in ranked:
        if len(chosen) >= target:
            break
        if a.id not in chosen:
            chosen.append(a.id)
            code_reasons[a.id] = "Added to complete the plan; a weaker match for your interests."
            topped_up += 1
    if topped_up and status == "success":
        warnings.append(f"Topped up {topped_up} activities by interest score.")

    # 5b. Guarantee some free options
    free_missing = MIN_FREE - sum(1 for i in chosen if by_id[i].price_per_person == 0)
    for a in ranked:
        if free_missing <= 0:
            break
        if a.price_per_person == 0 and a.id not in chosen:
            chosen.append(a.id)
            code_reasons[a.id] = "Added as a free option in case the budget is tight."
            free_missing -= 1

    # 6. Attach reasons to the real provider data.
    # Priority: the LLM's reason, then a factual interest match, then why code added it.
    candidates = []
    for activity_id in chosen:
        activity = by_id[activity_id]
        matched = _matched_interests(activity, interests)
        reason = (
            reasons.get(activity_id)
            or (f"Matches your interests: {', '.join(matched)}." if matched else None)
            or code_reasons.get(activity_id)
        )
        candidates.append(activity.model_copy(update={"reason": reason}))

    assumptions = []
    if not interests:
        assumptions.append("No interests given, so activities are a balanced first-visit mix.")

    summary = f"{len(candidates)} of {len(options)} open activities shortlisted."
    if note:
        summary += f" {note}"
    return {
        "activity_candidates": candidates,
        "trace": [
            trace_entry(AGENT, status, timer, summary, llm_result,
                        assumptions=assumptions, warnings=warnings, revision=revision)
        ],
    }