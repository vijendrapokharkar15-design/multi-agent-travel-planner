"""Streamlit front end for the Multi-Agent Travel Planner.

Run locally:  streamlit run app.py
"""

from datetime import date, timedelta

import streamlit as st
from pydantic import ValidationError

from agents.llm import LLMError, OpenAILLM
from graph.build import RECURSION_LIMIT, build_graph, initial_state
from schemas.models import MAX_TRIP_DAYS, TripRequest
from ui.render import render_results, status_line

MAX_PLANS_PER_SESSION = 5
DESTINATIONS = ["Lisbon", "Barcelona"]
INTERESTS = ["architecture", "art", "culture", "food", "history", "nature", "nightlife", "views"]

st.set_page_config(page_title="Multi-Agent Travel Planner", page_icon="✈️", layout="wide")
st.session_state.setdefault("plans_used", 0)
st.session_state.setdefault("result", None)


@st.cache_resource
def get_graph():
    """Built once per server process, not on every click."""
    return build_graph(OpenAILLM())


def run_plan(request: TripRequest | None = None, raw_request: str | None = None) -> None:
    if st.session_state.plans_used >= MAX_PLANS_PER_SESSION:
        st.warning(f"This demo allows {MAX_PLANS_PER_SESSION} plans per session. "
                   "Refresh the page later to try again.")
        return
    try:
        graph = get_graph()
    except LLMError as error:
        st.error(f"The app isn't configured yet: {error}")
        return

    st.session_state.plans_used += 1
    final = None
    with st.status("The agents are planning your trip...", expanded=True) as box:
        try:
            for mode, chunk in graph.stream(
                initial_state(request, raw_request),
                config={"recursion_limit": RECURSION_LIMIT},
                stream_mode=["updates", "values"],
            ):
                if mode == "updates":  # {node_name: what that node changed}
                    for update in chunk.values():
                        for t in (update or {}).get("trace", []):
                            box.write(status_line(t.agent, t.summary, t.revision))
                else:  # the full state after each step; keep the latest
                    final = chunk
        except Exception as error:  # a demo should never show a raw traceback
            box.update(label="Something went wrong", state="error")
            st.error(f"Planning stopped unexpectedly: {error}")
            return

        status = (final or {}).get("status")
        label = {"valid": "Plan ready", "invalid": "Finished, with some issues"}.get(
            status, "Couldn't make a plan")
        box.update(label=label, state="error" if status == "failed" else "complete",
                   expanded=False)
    st.session_state.result = final


# ---------- Header ----------
st.title("Multi-Agent Travel Planner")
st.caption("Specialist AI agents for flights, activities, hotels and the itinerary, "
           "coordinated by a LangGraph workflow with a validator and a revision loop.")
st.info("Demo data: airline and hotel names and all prices are fictional; place names are real. "
        "Flights depart from London. Nothing is booked.")

# ---------- Input ----------
form_tab, text_tab = st.tabs(["Plan with a form", "Describe your trip"])

with form_tab:
    with st.form("trip_form"):
        c1, c2 = st.columns(2)
        destination = c1.selectbox("Destination", DESTINATIONS)
        c1.text_input("From", "London", disabled=True,
                      help="The demo data covers flights from London only.")
        start_default = date.today() + timedelta(days=14)
        dates = c2.date_input(
            f"Travel dates (up to {MAX_TRIP_DAYS} days)",
            value=(start_default, start_default + timedelta(days=3)),
            min_value=date.today() + timedelta(days=1),
        )
        travellers = c2.number_input("Travellers", min_value=1, max_value=8, value=2)

        c3, c4, c5 = st.columns(3)
        budget = c3.number_input("Budget for the whole trip (£, 0 = no limit)",
                                 min_value=0, max_value=20000, value=1500, step=100)
        style = c4.selectbox("Style", ["budget", "mid-range", "luxury"], index=1)
        pace = c5.selectbox("Pace", ["relaxed", "balanced", "packed"], index=1)
        interests = st.multiselect("Interests", INTERESTS, default=["food", "architecture"])
        submitted = st.form_submit_button("Plan my trip", type="primary")

    if submitted:
        if not isinstance(dates, tuple) or len(dates) != 2:
            st.error("Please pick both a start and an end date.")
        else:
            try:
                request = TripRequest(
                    origin="London", destination=destination,
                    start_date=dates[0], end_date=dates[1],
                    travellers=int(travellers), budget_amount=float(budget) or None,
                    style=style, pace=pace, interests=interests,
                )
            except ValidationError:
                st.error(f"Please check the dates: trips can be at most {MAX_TRIP_DAYS} days.")
            else:
                run_plan(request=request)

with text_tab:
    with st.form("text_form"):
        raw = st.text_area(
            "Describe your trip",
            placeholder="4 days in Lisbon from 10 October for 2 people, about £1,500, "
                        "we love food and history",
            height=100,
        )
        sent = st.form_submit_button("Plan my trip", type="primary")
    if sent and raw.strip():
        run_plan(raw_request=raw)

st.caption(f"{MAX_PLANS_PER_SESSION - st.session_state.plans_used} of "
           f"{MAX_PLANS_PER_SESSION} plans left in this session.")

# ---------- Results ----------
result = st.session_state.result
if result:
    st.divider()
    if result.get("status") == "failed" and result.get("request") is None:
        # The parser stopped early with a question for the user
        st.info(result["warnings"][0] if result.get("warnings") else "Please add more detail.")
    else:
        render_results(result)