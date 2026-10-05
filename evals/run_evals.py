"""Evaluation runner: both systems, every case, several repeats, one scorer.

Usage (from the project root):
  python -m evals.run_evals                  # 8 cases x 3 repeats x 2 systems
  python -m evals.run_evals --repeats 1      # quick smoke run
  python -m evals.run_evals --cases "05,06"  # only some cases (quotes needed in PowerShell)

Writes:
  evals/results/runs/*.json   full result of every run (ignored by git)
  evals/results/results.csv   one row of metrics per run
  evals/results/summary.md    the tables for the README
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from agents.llm import OpenAILLM
from evals.baseline import run_baseline
from evals.cases import CASES, EVAL_TODAY, EvalCase
from graph.build import plan_trip

RESULTS_DIR = Path("evals/results")
RUNS_DIR = RESULTS_DIR / "runs"

# Mistakes that would ruin a real trip. Budget, thin days and format slips are scored separately.
HARD_CODES = {
    "activity_outside_trip", "activity_before_landing", "activity_after_departure",
    "activity_on_closed_day", "activity_outside_hours", "overlapping_items",
    "empty_day", "unknown_activity", "stay_dates_mismatch", "no_late_checkin",
}
PARSE_FIELDS = ("destination", "start_date", "end_date", "travellers", "budget_amount")


# ---------- Running each system ----------
def _selected(options, selected_id):
    return next((o for o in options or [] if o.id == selected_id), None)


def _norm(value):
    return value.lower() if isinstance(value, str) else value


def run_multi_agent(case: EvalCase, llm) -> dict:
    started = time.perf_counter()
    if case.raw_text:
        state = plan_trip(llm, raw_request=case.raw_text, today=EVAL_TODAY)
    else:
        state = plan_trip(llm, request=case.request, today=EVAL_TODAY)

    errors = []
    request = state.get("request")
    if case.raw_text:  # the structured request is the answer key for parsing
        if request is None:
            errors.append("Free text was not understood.")
        else:
            for field in PARSE_FIELDS:
                got, want = getattr(request, field), getattr(case.request, field)
                if _norm(got) != _norm(want):
                    errors.append(f"Parsed {field} = {got!r}, expected {want!r}.")

    trace = state.get("trace", [])
    return {
        "system": "multi_agent",
        "status": state.get("status"),
        "request": request or case.request,
        "flight": _selected(state.get("flight_options"), state.get("selected_flight_id")),
        "stay": _selected(state.get("stay_options"), state.get("selected_stay_id")),
        "itinerary": state.get("itinerary", []),
        "budget": state.get("budget"),
        "violations": state.get("violations", []),
        "errors": errors,
        "format_issues": [],
        "notes": " ".join(state.get("warnings", [])),
        "revisions": state.get("revision_count", 0),
        "llm_calls": sum(t.llm_calls for t in trace),
        "tokens": sum(t.tokens for t in trace),
        "cost_usd": sum(t.cost_usd for t in trace),
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


# ---------- Scoring (identical for both systems) ----------
def score(result: dict, case: EvalCase) -> dict:
    codes = [v.code for v in result["violations"]]
    hard = sum(1 for c in codes if c in HARD_CODES) + len(result["errors"])
    budget = result["budget"]
    if case.expect_unaffordable:
        # Success means an honest "no": not "valid", nothing broken, and the problem
        # either reported as over budget or declared infeasible.
        honest_no = "over_budget" in codes or result["status"] == "infeasible"
        success = result["status"] != "valid" and hard == 0 and honest_no
    else:
        success = result["status"] == "valid"
    return {
        "case": case.id,
        "system": result["system"],
        "status": result["status"],
        "success": success,
        "hard_violations": hard,
        "format_issues": len(result.get("format_issues", [])),
        "within_budget": bool(budget) and budget.over_by == 0,
        "thin_days": codes.count("thin_plan"),
        "activities": sum(1 for d in result["itinerary"] for i in d.items if i.kind == "activity"),
        "revisions": result.get("revisions", 0),
        "llm_calls": result["llm_calls"],
        "tokens": result["tokens"],
        "cost_usd": round(result["cost_usd"], 5),
        "latency_s": round(result["latency_ms"] / 1000, 1),
        "codes": ",".join(sorted(set(codes))),
        "errors": " | ".join(result["errors"]),
        "notes": result.get("notes", ""),
    }


# ---------- One job: run, save, score ----------
def _to_json(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else str(value)


def run_job(system: str, case: EvalCase, repeat: int, llm) -> dict:
    started = time.perf_counter()
    try:
        result = (run_multi_agent(case, llm) if system == "multi_agent"
                  else run_baseline(case.request, llm))
    except Exception as error:  # one broken run must not stop the whole eval
        result = {
            "system": system, "status": "crashed", "request": case.request,
            "flight": None, "stay": None, "itinerary": [], "budget": None,
            "violations": [], "errors": [f"Crashed: {error!r}"], "format_issues": [],
            "notes": "", "revisions": 0, "llm_calls": 0, "tokens": 0, "cost_usd": 0.0,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    saved = {k: v for k, v in result.items() if k != "activities"}  # keep files small
    path = RUNS_DIR / f"{system}_{case.id}_r{repeat}.json"
    path.write_text(json.dumps(saved, default=_to_json, indent=2), encoding="utf-8")
    return score(result, case)


# ---------- Summaries ----------
def _md_table(df: pd.DataFrame) -> str:
    """A small markdown table writer, to avoid an extra dependency."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def summarise(df: pd.DataFrame, repeats: int) -> str:
    pct = lambda s: f"{s.mean():.0%}"
    overall = df.groupby("system").agg(
        success=("success", pct),
        runs_with_hard_violations=("hard_violations", lambda s: f"{(s > 0).mean():.0%}"),
        hard_violations_per_run=("hard_violations", lambda s: f"{s.mean():.2f}"),
        format_issues_per_run=("format_issues", lambda s: f"{s.mean():.2f}"),
        said_infeasible=("status", lambda s: f"{(s == 'infeasible').mean():.0%}"),
        within_budget=("within_budget", pct),
        avg_activities=("activities", lambda s: f"{s.mean():.1f}"),
        avg_revisions=("revisions", lambda s: f"{s.mean():.2f}"),
        avg_llm_calls=("llm_calls", lambda s: f"{s.mean():.1f}"),
        median_latency_s=("latency_s", lambda s: f"{s.median():.1f}"),
        avg_cost_usd=("cost_usd", lambda s: f"{s.mean():.4f}"),
    ).reset_index()

    per_case = df.pivot_table(index="case", columns="system", values="success", aggfunc="sum")
    per_case = per_case.apply(lambda col: col.map(lambda n: f"{int(n)}/{repeats}")).reset_index()

    return "\n\n".join([
        "# Evaluation results",
        f"Run on {datetime.now():%Y-%m-%d %H:%M} with model `{os.getenv('OPENAI_MODEL')}`, "
        f"{len(df['case'].unique())} cases x {repeats} repeats per system. "
        "Both systems use the same model, the same mock data and the same deterministic scorer.",
        "## Overall",
        _md_table(overall),
        "## Successful runs per case",
        _md_table(per_case),
    ]) + "\n"


# ---------- Entry point ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the travel planner evaluation.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", default="", help='comma-separated case prefixes, e.g. "05,06"')
    args = parser.parse_args()

    cases = [c for c in CASES
             if not args.cases or any(c.id.startswith(p) for p in args.cases.split(","))]
    if not cases:
        parser.error(f"No cases match {args.cases!r}. In PowerShell, put the list in quotes.")
    jobs = [(system, case, r) for case in cases for r in range(1, args.repeats + 1)
            for system in ("multi_agent", "baseline")]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    llm = OpenAILLM()

    rows, started = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, s, c, r, llm) for s, c, r in jobs]
        for n, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{n}/{len(jobs)}] {row['system']:11} {row['case']:22} "
                  f"{row['status']:10} hard={row['hard_violations']} {row['latency_s']}s")

    df = pd.DataFrame(rows).sort_values(["case", "system"])
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    summary = summarise(df, args.repeats)
    (RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Finished {len(jobs)} runs in {time.perf_counter() - started:.0f} s.")


if __name__ == "__main__":
    main()