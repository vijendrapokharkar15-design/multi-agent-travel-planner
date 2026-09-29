"""Budget Advisor.

Runs only when the plan is over budget. Code works out which fixes are
possible and what each would REALISTICALLY save (facts); the LLM picks ONE
fix, weighing what the traveller would lose. The router then sends work back
to the agent responsible for that fix.
"""

from pydantic import BaseModel

from agents.common import Timer, trace_entry
from agents.flight import usable_hours
from agents.llm import LLM, LLMError
from core.budget import CONTINGENCY_RATE, rooms_needed, scheduled_activity_ids
from schemas.state import BudgetAction, BudgetAdvice, TripState

AGENT = "budget_advisor"

# When the LLM can't decide: the first fix in this order that closes the gap on
# its own, otherwise the largest saving. Activities come last because removing
# them makes the trip emptier.
PREFERENCE: list[BudgetAction] = ["cheaper_flight", "cheaper_stay", "drop_paid_activities"]


# ---------- What the LLM is allowed to return ----------
class AdvisorDecision(BaseModel):
    action: BudgetAction
    reason: str


SYSTEM_PROMPT = """You are the Budget Advisor in a travel planner. The plan is over budget.
Choose ONE fix from the possible actions listed. Each shows a realistic estimated saving.
Prefer the fix that closes the gap while losing the least of what matters to this traveller.
Prefer changing the flight or the hotel over removing activities, because removing
activities leaves the trip emptier. Only prefer removing activities when the traveller's
style is budget, or when no other fix comes close.
reason: one short sentence the traveller will read.
Only choose an action from the list."""


# ---------- Deterministic helpers ----------
def _selected(options: list, selected_id: str | None):
    return next((o for o in options if o.id == selected_id), None)


def possible_fixes(state: TripState) -> dict[BudgetAction, tuple[float, str]]:
    """Every fix that could reduce the budgeted total, with a realistic saving
    (including contingency) and a short description. Pure code."""
    request = state["request"]
    uplift = 1 + CONTINGENCY_RATE
    over_by = state["budget"].over_by if state.get("budget") else 0
    fixes: dict[BudgetAction, tuple[float, str]] = {}

    # Cheaper flight: the cheaper option that keeps the most usable time
    flight = _selected(state.get("flight_options", []), state.get("selected_flight_id"))
    if flight and request.budget_includes_flights:
        cheaper = [f for f in state.get("flight_options", [])
                   if f.price_per_person < flight.price_per_person]
        if cheaper:
            best = max(cheaper, key=lambda f: (usable_hours(request, f), -f.price_per_person))
            saving = (flight.price_per_person - best.price_per_person) * request.travellers * uplift
            fixes["cheaper_flight"] = (
                round(saving, 2),
                f"switch to {best.airline}: {usable_hours(request, best)} usable hours "
                f"instead of {usable_hours(request, flight)}",
            )

    # Cheaper stay
    stay = _selected(state.get("stay_options", []), state.get("selected_stay_id"))
    if stay and request.budget_includes_stay and request.num_nights > 0:
        cheaper = [s for s in state.get("stay_options", [])
                   if s.price_per_night < stay.price_per_night]
        if cheaper:
            best = min(cheaper, key=lambda s: s.price_per_night)
            saving = ((stay.price_per_night - best.price_per_night)
                      * request.num_nights * rooms_needed(request.travellers) * uplift)
            fixes["cheaper_stay"] = (round(saving, 2), f"switch hotel {stay.name} to {best.name}")

    # Drop paid activities: only as many as needed, most expensive first
    by_id = {a.id: a for a in state.get("activity_candidates", [])}
    paid = sorted(
        (by_id[i] for i in scheduled_activity_ids(state.get("itinerary", []))
         if i in by_id and by_id[i].price_per_person > 0),
        key=lambda a: a.price_per_person, reverse=True,
    )
    if paid:
        saving, dropped = 0.0, 0
        for activity in paid:
            if saving >= over_by:
                break
            saving += activity.price_per_person * request.travellers * uplift
            dropped += 1
        fixes["drop_paid_activities"] = (
            round(saving, 2),
            f"drop {dropped} of {len(paid)} paid activities, most expensive first",
        )
    return fixes


def fallback_action(fixes: dict[BudgetAction, tuple[float, str]], over_by: float) -> BudgetAction:
    for action in PREFERENCE:
        if action in fixes and fixes[action][0] >= over_by:
            return action
    return max(fixes, key=lambda a: fixes[a][0])


# ---------- The node ----------
def run_budget_advisor(state: TripState, llm: LLM) -> dict:
    timer = Timer()
    request = state["request"]
    budget = state["budget"]
    revision = state.get("revision_count", 0)
    currency = budget.currency

    fixes = possible_fixes(state)
    if not fixes:
        advice = BudgetAdvice(
            action=None,
            reason="No cheaper flights, hotels or paid activities are left to cut.",
        )
        return {
            "budget_advice": advice,
            "trace": [trace_entry(AGENT, "success", timer,
                                  f"Over by {currency} {budget.over_by:.2f}; no fix available.",
                                  revision=revision)],
        }

    options_text = "\n".join(
        f"- {action}: saves up to {currency} {saving:.2f} ({description})"
        for action, (saving, description) in fixes.items()
    )
    user_prompt = (
        f"Over budget by {currency} {budget.over_by:.2f} "
        f"(total {currency} {budget.total:.2f}, budget {currency} {budget.budget_amount:.2f}).\n"
        f"Traveller: interests={request.interests or 'none given'}, style={request.style}, "
        f"pace={request.pace}, travellers={request.travellers}.\n"
        f"Possible actions:\n{options_text}"
    )

    status, warnings, llm_result = "success", [], None
    try:
        llm_result = llm.structured(SYSTEM_PROMPT, user_prompt, AdvisorDecision)
        action, reason = llm_result.parsed.action, llm_result.parsed.reason
        if action not in fixes:
            raise LLMError(f"LLM chose '{action}', which is not possible here")
    except LLMError as error:
        status = "partial"
        warnings.append(f"Used fallback rule (preferred fix that closes the gap): {error}")
        action = fallback_action(fixes, budget.over_by)
        reason = f"Largest available saving ({currency} {fixes[action][0]:.2f})." \
            if fixes[action][0] < budget.over_by \
            else f"Closes the gap with a saving of {currency} {fixes[action][0]:.2f}."

    saving = fixes[action][0]
    advice = BudgetAdvice(action=action, reason=reason, estimated_saving=saving)
    summary = (
        f"Over by {currency} {budget.over_by:.2f}; chose {action} "
        f"(saves up to {currency} {saving:.2f})."
    )
    return {
        "budget_advice": advice,
        "trace": [trace_entry(AGENT, status, timer, summary, llm_result,
                              warnings=warnings, revision=revision)],
    }