# AI_Scenario_Sim/engine/event_validator.py
# -----------------------------------------
"""
event_validator.py

Performs fully exhaustive checks on data/events_catalogue.json:
  1) Unique event IDs
  2) Valid stage names
  3) Non-negative integer year_offset
  4) condition references exist & come from same/earlier stage
  5) next_events references exist & point to same/later stage
  6) next_events probabilities sum to 1 (±1e-6)
  7) No cycles in the directed graph of dependencies
  8) All events are reachable from at least one root ("ALWAYS")
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Set

# Allowed stage names and their ordering index
STAGE_ORDER = {
    "Pre-AGI": 0,
    "AGI-Rollout": 1,
    "Self-Improving": 2,
    "Post-ASI": 3
}

def load_events(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def validate_catalogue(events: List[Dict]) -> None:
    errors: List[str] = []
    all_ids: Set[str]   = set()
    id_to_stage: Dict[str, str] = {}
    id_to_index: Dict[str, int] = {}

    # 1) Check for duplicate IDs and valid stage names / year_offset
    for idx, ev in enumerate(events):
        eid = ev.get("id")
        if not isinstance(eid, str) or not eid:
            errors.append(f"[EVENT {idx}] Missing or invalid 'id'.")
            continue

        # Duplicate ID check
        if eid in all_ids:
            errors.append(f"Duplicate event ID found: '{eid}'.")
        else:
            all_ids.add(eid)
            id_to_index[eid] = idx

        # Check stage validity
        stage = ev.get("stage")
        if stage not in STAGE_ORDER:
            errors.append(f"Event '{eid}' has invalid stage '{stage}'. "
                          f"Must be one of {list(STAGE_ORDER.keys())}.")

        # Check year_offset is non-negative integer
        yo = ev.get("year_offset")
        if not isinstance(yo, int) or yo < 0:
            errors.append(f"Event '{eid}' has invalid 'year_offset': {yo}. "
                          f"Must be an integer ≥ 0.")

        # Record stage for later checks
        id_to_stage[eid] = stage

    # 2) Verify condition references exist & are from same/earlier stage
    for ev in events:
        eid = ev["id"]
        conds = ev.get("condition", [])
        # condition must be a non-empty list
        if not isinstance(conds, list) or len(conds) == 0:
            errors.append(f"Event '{eid}' has invalid 'condition' field: {conds}.")
            continue

        # If condition is exactly ["ALWAYS"], skip further checks
        if conds == ["ALWAYS"]:
            continue

        # Otherwise, each condition must reference an existing ID and be from same or earlier stage
        for c in conds:
            if c not in all_ids:
                errors.append(f"Event '{eid}' condition references missing ID '{c}'.")
            else:
                # Stage ordering check
                parent_stage = id_to_stage.get(c)
                child_stage  = id_to_stage.get(eid)
                if STAGE_ORDER[parent_stage] > STAGE_ORDER[child_stage]:
                    errors.append(
                        f"Event '{eid}' is in stage '{child_stage}' but its condition '{c}' "
                        f"is in later stage '{parent_stage}'."
                    )

    # 3) Verify next_events references exist & are from same/later stage, and probabilities sum to 1
    for ev in events:
        eid = ev["id"]
        nxt = ev.get("next_events", [])
        if not isinstance(nxt, list):
            errors.append(f"Event '{eid}' has invalid 'next_events' (not a list).")
            continue

        if nxt:
            total_p = 0.0
            for child_id, p in nxt:
                # Existence check
                if child_id not in all_ids:
                    errors.append(f"Event '{eid}' next_events references missing ID '{child_id}'.")
                    continue

                # Stage ordering: child must be in same or a later stage
                parent_stage = id_to_stage.get(eid)
                child_stage  = id_to_stage.get(child_id)
                if STAGE_ORDER[child_stage] < STAGE_ORDER[parent_stage]:
                    errors.append(
                        f"Event '{eid}' (stage '{parent_stage}') lists child '{child_id}' "
                        f"in earlier stage '{child_stage}'."
                    )

                # Probability validity
                if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
                    errors.append(
                        f"Event '{eid}' next_events probability for '{child_id}' "
                        f"is invalid: {p}."
                    )
                else:
                    total_p += float(p)

            # Sum-to-1 check (±1e-6)
            if abs(total_p - 1.0) > 1e-6:
                errors.append(
                    f"Event '{eid}' next_events probabilities sum to {total_p:.6f} (must be 1.0)."
                )

    # 4) Build directed graph of all dependencies (edges for both condition and next_events)
    graph: Dict[str, List[str]] = {eid: [] for eid in all_ids}
    for ev in events:
        eid = ev["id"]
        # Edges from prerequisite → event
        for c in ev.get("condition", []):
            if c != "ALWAYS" and c in all_ids:
                graph[c].append(eid)
        # Edges from event → child
        for child_id, _ in ev.get("next_events", []):
            if child_id in all_ids:
                graph[eid].append(child_id)

    # 5) Detect cycles using DFS
    visited: Set[str] = set()
    on_stack: Set[str] = set()
    cycle_found = False

    def dfs_cycle(node: str):
        nonlocal cycle_found
        if cycle_found:
            return
        visited.add(node)
        on_stack.add(node)
        for nbr in graph.get(node, []):
            if nbr not in visited:
                dfs_cycle(nbr)
            elif nbr in on_stack:
                errors.append(f"Cycle detected: '{nbr}' is in the recursion stack.")
                cycle_found = True
                return
        on_stack.remove(node)

    for eid in all_ids:
        if eid not in visited:
            dfs_cycle(eid)
            if cycle_found:
                break

    # 6) Ensure every non-root event is reachable from at least one root
    #    Roots = events with condition == ["ALWAYS"]
    roots = {ev["id"] for ev in events if ev.get("condition") == ["ALWAYS"]}
    if not roots:
        errors.append("No root events found (condition == ['ALWAYS']).")

    reachable: Set[str] = set()

    # BFS/DFS from each root to collect reachable event IDs
    def dfs_reach(node: str):
        if node in reachable:
            return
        reachable.add(node)
        for nbr in graph.get(node, []):
            dfs_reach(nbr)

    for r in roots:
        dfs_reach(r)

    for eid in all_ids:
        # Every event must be either a root or reachable from a root
        if eid not in reachable:
            errors.append(f"Event '{eid}' is unreachable from any root.")

    # 7) Final report
    if errors:
        print("Validation errors in events_catalogue.json:")
        for err in errors:
            print("  •", err)
        sys.exit(1)
    else:
        print("events_catalogue.json passed all exhaustive checks ✅")


if __name__ == "__main__":
    path = Path(__file__).parents[1] / "data" / "events_catalogue.json"
    evts = load_events(str(path))
    validate_catalogue(evts)
