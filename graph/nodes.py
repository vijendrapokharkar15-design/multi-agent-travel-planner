"""The code nodes of the graph: budget, validate, router and compile,
plus the small functions that choose the next node. No LLM calls here."""

from langgraph.graph import END

from agents.common import Timer, trace_entry
from core.budget import CurrencyMismatchError, compute_budget
from core.validator import VIOLATION_OWNER, validate_plan
from schemas.state import (
    MAX_REVISIONS,
    BudgetAction,
    NodeName,
    RevisionRequest,
    TripState,
    Violation,
)

UPSTREAM_ORDER: list[NodeName] = ["flight_agent", "stay_agent", "itinerary_agent"]
ACTION_TARGET: dict[BudgetAction, NodeName] = {
    "cheaper_flight": "flight_agent",
    "cheaper_stay": "stay_agent",
    "drop_paid_activities": "itinerary_agent",
}
FATAL_IF_FAILED = {"flight_agent", "stay_agent"}  # no plan is possible without these


# ---------- Helpers ----------
def _selected(options: list, selected_id: str | None):
    return next((o for o in options if o.id == selected_id), None)


def _plan_parts(state: TripState):
    return (
        state["request"],
        _selected(state.get("flight_options", []), state.get("selected_flight_id")),
        _selected(state.get("stay_options", []), state.get("selected_stay_id")),
        state.get("itinerary", []),
        state.get("activity_candidates", []),
    )


# ---------- Nodes ----------
def budget_node(state: TripState) -> dict:
    timer = Timer()
    revision = state.get("revision_count", 0)
    request, flight, stay, itinerary, activities = _plan_parts(state)
    try:
        budget = compute_budget(request, flight, stay, itinerary, activities)
    except (CurrencyMismatchError, ValueError) as error:
        return {
            "budget": None,
            "budget_advice": None,
            "trace": [trace_entry("budget", "failed", timer, str(error), revision=revision)],
        }

    summary = f"Total {budget.currency} {budget.total:,.2f}"
    if budget.over_by:
        summary += f", over budget by {budget.over_by:,.2f}."
    elif budget.budget_amount:
        summary += f", within the budget of {budget.budget_amount:,.2f}."
    else:
        summary += "."
    return {
        "budget": budget,
        "budget_advice": None,  # cleared each time; the advisor sets it only when over budget
        "trace": [trace_entry("budget", "success", timer, summary, revision=revision)],
    }


def validate_node(state: TripState) -> dict:
    timer = Timer()
    request, flight, stay, itinerary, activities = _plan_parts(state)
    violations = validate_plan(request, flight, stay, itinerary, activities, state.get("budget"))
    summary = (
        "No violations." if not violations
        else f"{len(violations)} violation(s): {', '.join(v.code for v in violations)}."
    )
    return {
        "violations": violations,
        "trace": [trace_entry("validate", "success", timer, summary,
                              revision=state.get("revision_count", 0))],
    }


def router_node(state: TripState) -> dict:
    """Decides whether the plan is finished or which agent must revise it."""
    timer = Timer()
    count = state.get("revision_count", 0)
    violations: list[Violation] = state.get("violations", [])

    def stop(status: str, summary: str, warnings: list[str] | None = None) -> dict:
        update = {
            "status": status,
            "revision_request": None,
            "trace": [trace_entry("router", "success" if status == "valid" else "partial",
                                  timer, summary, revision=count)],
        }
        if warnings:
            update["warnings"] = warnings
        return update

    # 1. A critical agent failed in this round: no plan is possible
    failed = [
        t.agent for t in state.get("trace", [])
        if t.agent in FATAL_IF_FAILED and t.status == "failed" and t.revision == count
    ]
    if failed:
        return stop("failed", f"Stopped: {', '.join(failed)} could not complete.")

    # 2. Nothing wrong: done
    if not violations:
        return stop("valid", "Plan is valid." if count == 0
                    else f"Plan is valid after {count} revision(s).")

    # 3. Out of revisions: return the best plan with its problems shown
    messages = [v.message for v in violations]
    if count >= MAX_REVISIONS:
        return stop("invalid", f"Stopped after {MAX_REVISIONS} revisions; "
                               f"{len(violations)} problem(s) remain.", messages)

    # 4. Work out who can fix what
    advice = state.get("budget_advice")
    by_target: dict[NodeName, list[Violation]] = {}
    for v in violations:
        owner = VIOLATION_OWNER[v.code]
        if owner == "budget_advisor":
            owner = ACTION_TARGET[advice.action] if advice and advice.action else None
        if owner:
            by_target.setdefault(owner, []).append(v)
    if not by_target:
        return stop("invalid", "Remaining problems cannot be fixed by re-planning.", messages)

    # 5. Fix the most upstream problem first; downstream nodes rerun automatically
    target = next(n for n in UPSTREAM_ORDER if n in by_target)
    fixing = by_target[target]
    action = advice.action if advice and any(v.code == "over_budget" for v in fixing) else None

    excluded = []
    if target == "flight_agent" and state.get("selected_flight_id"):
        excluded.append(state["selected_flight_id"])
    if target == "stay_agent" and state.get("selected_stay_id"):
        excluded.append(state["selected_stay_id"])

    reason = " ".join(v.message for v in fixing)
    if action:
        reason += f" Suggested fix: {action} ({advice.reason})"

    return {
        "revision_request": RevisionRequest(
            target=target, codes=[v.code for v in fixing], reason=reason, budget_action=action
        ),
        "revision_count": count + 1,
        "excluded_ids": excluded,
        "status": "running",
        "trace": [trace_entry(
            "router", "success", timer,
            f"Revision {count + 1}: {', '.join(v.code for v in fixing)} sent to {target}"
            + (f" ({action})." if action else "."),
            revision=count,
        )],
    }


def compile_node(state: TripState) -> dict:
    timer = Timer()
    status = state.get("status", "invalid")
    budget = state.get("budget")
    llm_calls = sum(t.llm_calls for t in state.get("trace", []))
    cost = sum(t.cost_usd for t in state.get("trace", []))
    summary = (
        f"Final plan: {status} after {state.get('revision_count', 0)} revision(s)"
        + (f", total {budget.currency} {budget.total:,.2f}" if budget else "")
        + f". {llm_calls} LLM calls, about ${cost:.4f}."
    )
    return {"trace": [trace_entry("compile", "success", timer, summary,
                                  revision=state.get("revision_count", 0))]}


# ---------- Choosing the next node (used by conditional edges) ----------
def after_parse(state: TripState):
    """Stop with the question, or start both agents in parallel."""
    if state.get("status") == "failed":
        return END
    return ["flight_agent", "activities_agent"]


def after_budget(state: TripState) -> str:
    budget = state.get("budget")
    return "budget_advisor" if budget and budget.over_by > 0 else "validate"


def after_router(state: TripState) -> str:
    if state.get("status") in ("valid", "invalid", "failed"):
        return "compile"
    return state["revision_request"].target