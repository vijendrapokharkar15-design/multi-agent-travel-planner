"""Wires every node into one LangGraph StateGraph, and provides plan_trip()
to run a whole planning request from start to finish."""

from datetime import date

from langgraph.graph import END, START, StateGraph

from agents.activities import run_activities_agent
from agents.budget_advisor import run_budget_advisor
from agents.flight import run_flight_agent
from agents.itinerary import run_itinerary_agent
from agents.llm import LLM
from agents.parser import run_parser
from agents.stay import run_stay_agent
from graph.nodes import (
    after_budget,
    after_parse,
    after_router,
    budget_node,
    compile_node,
    router_node,
    validate_node,
)
from providers.base import ActivityProvider, FlightProvider, StayProvider
from providers.mock import MockActivityProvider, MockFlightProvider, MockStayProvider
from schemas.models import TripRequest
from schemas.state import TripState

# LangGraph's safety net against endless loops. MAX_REVISIONS (3) is our real cap;
# this only needs to be comfortably above the steps 3 revisions can take.
RECURSION_LIMIT = 60


def build_graph(
    llm: LLM,
    flights: FlightProvider | None = None,
    stays: StayProvider | None = None,
    activities: ActivityProvider | None = None,
    today: date | None = None,
):
    """Dependencies are passed in, so tests can use FakeLLM and real runs OpenAILLM."""
    flights = flights or MockFlightProvider()
    stays = stays or MockStayProvider()
    activities = activities or MockActivityProvider()

    g = StateGraph(TripState)

    # Nodes: each lambda binds the node's dependencies, so LangGraph only passes state
    g.add_node("parse", lambda s: run_parser(s, llm, today))
    g.add_node("flight_agent", lambda s: run_flight_agent(s, flights, llm))
    g.add_node("activities_agent", lambda s: run_activities_agent(s, activities, llm))
    g.add_node("stay_agent", lambda s: run_stay_agent(s, stays, llm))
    g.add_node("itinerary_agent", lambda s: run_itinerary_agent(s, llm))
    g.add_node("budget", budget_node)
    g.add_node("budget_advisor", lambda s: run_budget_advisor(s, llm))
    g.add_node("validate", validate_node)
    g.add_node("router", router_node)
    g.add_node("compile", compile_node)

    # Edges
    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", after_parse, ["flight_agent", "activities_agent", END])
    g.add_edge("flight_agent", "stay_agent")  # two plain edges, not a wait-for-both join,
    g.add_edge("activities_agent", "stay_agent")  # so a flight-only revision still reaches Stay
    g.add_edge("stay_agent", "itinerary_agent")
    g.add_edge("itinerary_agent", "budget")
    g.add_conditional_edges("budget", after_budget, ["budget_advisor", "validate"])
    g.add_edge("budget_advisor", "validate")
    g.add_edge("validate", "router")
    g.add_conditional_edges(
        "router", after_router, ["flight_agent", "stay_agent", "itinerary_agent", "compile"]
    )
    g.add_edge("compile", END)

    return g.compile()


def initial_state(request: TripRequest | None = None, raw_request: str | None = None) -> TripState:
    if request is None and not raw_request:
        raise ValueError("Give either a structured request or free text.")
    state: TripState = {
        "revision_count": 0,
        "status": "running",
        "excluded_ids": [],
        "trace": [],
        "warnings": [],
    }
    if request is not None:
        state["request"] = request
    if raw_request:
        state["raw_request"] = raw_request
    return state


def plan_trip(
    llm: LLM,
    request: TripRequest | None = None,
    raw_request: str | None = None,
    **dependencies,
) -> TripState:
    """Run one planning request end to end and return the final state."""
    graph = build_graph(llm, **dependencies)
    return graph.invoke(
        initial_state(request, raw_request),
        config={"recursion_limit": RECURSION_LIMIT},
    )