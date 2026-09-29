"""Functions that draw each part of the result. app.py calls these;
nothing here runs the graph or calls the LLM."""

import pandas as pd
import streamlit as st

from agents.flight import usable_hours
from core.budget import rooms_needed
from schemas.state import TripState

AGENT_LABELS = {
    "parser": "Request parser",
    "flight_agent": "Flight Agent",
    "activities_agent": "Activities Agent",
    "stay_agent": "Stay Agent",
    "itinerary_agent": "Itinerary Agent",
    "budget": "Budget",
    "budget_advisor": "Budget Advisor",
    "validate": "Validator",
    "router": "Router",
    "compile": "Final check",
}


# ---------- Helpers ----------
def _selected(options, selected_id):
    return next((o for o in options or [] if o.id == selected_id), None)


def _gbp(amount: float) -> str:
    return f"£{amount:,.2f}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _labels(labels: list[str]) -> str:
    return " · ".join(label.replace("_", " ").capitalize() for label in labels)


def status_line(agent: str, summary: str, revision: int) -> str:
    """One line for the live status panel."""
    prefix = f"[revision {revision}] " if revision else ""
    return f"{prefix}**{AGENT_LABELS.get(agent, agent)}**: {summary}"


def _budgeted_total(state: TripState) -> float:
    """The part of the total that counts against the budget (respects the scope flags)."""
    request, budget = state["request"], state["budget"]
    counted = budget.total
    if not request.budget_includes_flights:
        counted -= budget.flights
    if not request.budget_includes_stay:
        counted -= budget.stay
    return counted


# ---------- Tabs ----------
def render_overview(state: TripState) -> None:
    status = state.get("status")
    revisions = state.get("revision_count", 0)
    if status == "valid":
        st.success("Plan passed every check"
                   + (f" after {_plural(revisions, 'revision')}." if revisions else "."))
    elif status == "invalid":
        st.warning("Here's the best plan found, but some problems remain (see below).")
    else:
        st.error("A plan could not be made for this request.")

    budget = state.get("budget")
    if budget:
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated total", _gbp(budget.total))
        c2.metric("Per traveller", _gbp(budget.per_traveller))
        if budget.budget_amount:
            c3.metric("Budget", _gbp(budget.budget_amount))
            if budget.over_by:
                c3.markdown(f":red[{_gbp(budget.over_by)} over budget]")
            else:
                under = budget.budget_amount - _budgeted_total(state)
                c3.markdown(f":green[{_gbp(under)} under budget]")

    flight = _selected(state.get("flight_options"), state.get("selected_flight_id"))
    stay = _selected(state.get("stay_options"), state.get("selected_stay_id"))
    if flight:
        st.markdown(f"**Flight:** {flight.airline}, lands {flight.outbound_arrive:%a %H:%M}, "
                    f"returns {flight.return_depart:%a %H:%M} "
                    f"({usable_hours(state['request'], flight)} usable hours at the destination)")
    if stay:
        st.markdown(f"**Stay:** {stay.name}, {stay.area}")

    for warning in state.get("warnings", []):
        st.warning(warning)


def render_flights(state: TripState) -> None:
    request = state["request"]
    selected_id = state.get("selected_flight_id")
    for f in state.get("flight_options", []):
        with st.container(border=True):
            title = f"{f.airline} ({f.outbound_flight_number} / {f.return_flight_number})"
            st.markdown(f"**{title}**" + ("  :green[Selected]" if f.id == selected_id else ""))
            if f.labels:
                st.caption(_labels(f.labels))
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"Out: {f.outbound_depart:%a %H:%M} to {f.outbound_arrive:%H:%M}  \n"
                        f"{_plural(f.outbound_stops, 'stop')}, {f.outbound_duration_mins} min")
            c2.markdown(f"Back: {f.return_depart:%a %H:%M} to {f.return_arrive:%H:%M}  \n"
                        f"{_plural(f.return_stops, 'stop')}, {f.return_duration_mins} min")
            c3.markdown(f"**{_gbp(f.price_per_person)}** per person  \n"
                        f"{usable_hours(request, f)} usable hours · source: {f.provenance}")
            if f.reason:
                st.write(f.reason)


def render_stay(state: TripState) -> None:
    request = state["request"]
    nights, rooms = request.num_nights, rooms_needed(request.travellers)
    options = state.get("stay_options", [])
    if not options:
        st.info("No accommodation needed for a same-day trip." if nights == 0
                else "No accommodation found.")
        return
    selected_id = state.get("selected_stay_id")
    for s in options:
        with st.container(border=True):
            st.markdown(f"**{s.name}**, {s.area}"
                        + ("  :green[Selected]" if s.id == selected_id else ""))
            if s.labels:
                st.caption(_labels(s.labels))
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"{_gbp(s.price_per_night)} per night  \n"
                        f"**{_gbp(s.total_price(nights) * rooms)}** for "
                        f"{_plural(nights, 'night')}, {_plural(rooms, 'room')}")
            c2.markdown(f"Rating: {s.rating if s.rating is not None else 'n/a'}  \n"
                        f"Late check-in: {'yes' if s.late_checkin_ok else 'no'}")
            c3.markdown(f"Source: {s.provenance}")
            if s.reason:
                st.write(s.reason)


def render_itinerary(state: TripState) -> None:
    activities = {a.id: a for a in state.get("activity_candidates", [])}
    for day in state.get("itinerary", []):
        st.subheader(f"{day.date:%A %d %B}" + (f": {day.notes}" if day.notes else ""))
        for item in day.items:
            when = f"{item.start_time:%H:%M} to {item.end_time:%H:%M}" if item.start_time else "Any time"
            line = f"`{when}`  {item.title}"
            activity = activities.get(item.activity_id) if item.activity_id else None
            if activity:
                extras = [f"{activity.area}"]
                extras.append("free" if activity.price_per_person == 0
                              else f"{_gbp(activity.price_per_person)} pp")
                if activity.booking_needed:
                    extras.append("book ahead")
                line += f"  ({', '.join(extras)})"
            elif item.kind in ("meal", "transfer"):
                line = f"`{when}`  _{item.title}_"
            st.markdown(line)


def render_budget(state: TripState) -> None:
    budget = state.get("budget")
    if not budget:
        st.info("No budget was calculated.")
        return
    rows = {
        "Flights": budget.flights,
        "Accommodation": budget.stay,
        "Activities": budget.activities,
        "Food (estimate)": budget.food_estimate,
        "Local transport (estimate)": budget.local_transport_estimate,
        "Contingency (10%)": budget.contingency,
    }
    table = pd.DataFrame({"Category": list(rows), "Amount (£)": list(rows.values())})
    st.bar_chart(table, x="Category", y="Amount (£)", horizontal=True)
    st.dataframe(table, hide_index=True, width="stretch")
    st.markdown(f"**Total: {_gbp(budget.total)}** ({_gbp(budget.per_traveller)} per traveller)")
    st.caption("Food, transport and contingency come from fixed, documented estimate rules, "
               "not live prices.")


def render_assumptions(state: TripState) -> None:
    assumptions = list(state["request"].assumptions)
    for t in state.get("trace", []):
        for a in t.assumptions:
            if a not in assumptions:
                assumptions.append(a)
    st.markdown("**Assumptions**")
    if assumptions:
        for a in assumptions:
            st.markdown(f"- {a}")
    else:
        st.markdown("- None: everything came from your request.")

    st.markdown("**About the data**")
    st.markdown(
        "- Flights, hotels and prices are **mock data** for a demo; airline and hotel names "
        "are fictional, place names are real.\n"
        "- Opening hours and closures are illustrative, not checked facts.\n"
        "- Distances are straight-line estimates, so real travel times may be longer.\n"
        "- Nothing is booked. Always check with the provider before travelling."
    )


def render_trace(state: TripState) -> None:
    trace = state.get("trace", [])
    c1, c2, c3 = st.columns(3)
    c1.metric("Revisions", state.get("revision_count", 0))
    c2.metric("LLM calls", sum(t.llm_calls for t in trace))
    c3.metric("Estimated LLM cost", f"${sum(t.cost_usd for t in trace):.4f}")

    table = pd.DataFrame([{
        "Round": t.revision,
        "Step": AGENT_LABELS.get(t.agent, t.agent),
        "Status": t.status,
        "Time (ms)": t.latency_ms,
        "Tokens": t.tokens,
        "Summary": t.summary,
    } for t in trace])
    # st.table wraps long text, so every summary is fully readable
    st.table(table.set_index("Round"))

    warnings = [(t.agent, w) for t in trace for w in t.warnings]
    if warnings:
        with st.expander(f"Developer notes ({len(warnings)})"):
            for agent, w in warnings:
                st.markdown(f"- **{AGENT_LABELS.get(agent, agent)}**: {w}")


def render_results(state: TripState) -> None:
    tabs = st.tabs(["Overview", "Flights", "Stay", "Itinerary", "Budget",
                    "Assumptions and data", "Agent trace"])
    renderers = [render_overview, render_flights, render_stay, render_itinerary,
                 render_budget, render_assumptions, render_trace]
    for tab, render in zip(tabs, renderers):
        with tab:
            render(state)