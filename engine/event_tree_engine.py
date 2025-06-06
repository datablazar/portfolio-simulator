# AI_Scenario_Sim/engine/event_tree_engine.py
# -------------------------------------------
"""
EventTreeEngine
===============
• Loads Event objects.
• Given a timeline (ai, agi, asi) and RNG, decides which events fire.
• Respects each event’s `condition` list (prerequisites).
• Returns:
    - fired: List[Event]       (events that actually fired)
    - drift: np.ndarray        (shape: horizon × n_assets) additive drifts
    - volmul: np.ndarray       (shape: horizon) multiplicative vol factors
    - triggered_ep3_event_id: str | None (NEW: ID of the last triggered Post-ASI event, if any)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import json
import numpy as np
from pathlib import Path
from .event_schema import Event

class EventTreeEngine:
    def __init__(self,
                 events_path: str | Path,
                 tickers: List[str],
                 horizon_years: int = 40,
                 seed: int | None = None):
        # Load all events from JSON
        self.events = self._load_events(events_path)
        self.tickers = tickers
        self.horizon = horizon_years
        # seed=None → nondeterministic RNG
        self.rng = np.random.default_rng(seed)

        # Map id → Event for quick lookup
        self._by_id: Dict[str, Event] = {e.id: e for e in self.events}

        # Identify root events (condition == ["ALWAYS"])
        self._root_events = [e for e in self.events if e.condition == ["ALWAYS"]]

        # NEW: Store the last triggered E-P3 event ID
        self.triggered_ep3_event_id: str | None = None

        # NEW: Load event_stats.json and create the map
        self.event_stats_map = self._load_event_stats_map()

        # Initialize per-run state
        self.reset()

    def reset(self) -> None:
        """Clear state so events are evaluated independently."""
        self._fired_events_history: set[str] = set()
        self.triggered_ep3_event_id = None

    def _load_events(self, events_path: str | Path) -> List[Event]:
        """Loads events from a JSON file."""
        with open(events_path, "r") as f:
            raw_events = json.load(f)
        return [Event(**data) for data in raw_events]

    def _load_event_stats_map(self) -> Dict[str, Any]: # Changed return type hint to Any for flexibility
        """Loads event statistics from event_stats.json and creates a map."""
        event_stats_path = Path(__file__).resolve().parent.parent / "data" / "event_stats.json"
        with open(event_stats_path, "r") as f:
            raw_stats = json.load(f)
        
        # Map id -> (drift_mu_dict, drift_sigma_dict, vol_mu_dict, vol_sigma_dict)
        parsed_map = {}
        for entry in raw_stats:
            parsed_map[entry["id"]] = (
                entry.get("drift_mu", {}),
                entry.get("drift_sigma", {}),
                entry.get("vol_mu", {}),
                entry.get("vol_sigma", {})
            )
        return parsed_map

    def simulate_events(self,
                        timeline: Dict[str, int | None],
                        current_year: int
                        ) -> Dict[str, Any]:
        """
        Simulates events for a given year based on timeline and current state.

        Args:
            timeline: Dictionary with 'ai_year', 'agi_year', 'asi_year'.
            current_year: The current year of the simulation.

        Returns:
            A dictionary containing:
                - "fired": List of Event objects that fired in the current year.
                - "drift": numpy array of additive drifts for assets.
                - "volmul": Multiplicative volatility factor.
                - "triggered_ep3_event_id": The ID of the last triggered E-P3 event, if any.
        """
        fired_events_this_year: List[Event] = []
        # Initialize drifts and volatility for this year
        year_drift = np.zeros(len(self.tickers))
        year_vol_multiplier_array = np.ones(len(self.tickers)) # Initialized as array of 1.0s

        # --- NEW: Check if an E-P3 event has already been persistently triggered ---
        if self.triggered_ep3_event_id is not None:
            # If an E-P3 event was triggered in a previous year and is meant to persist,
            # we simply return its status without processing new events,
            # as its impact is now handled by the macro state.
            return {
                "fired": [],
                "drift": year_drift,
                "volmul": year_vol_multiplier_array, # Return the default ones array
                "triggered_ep3_event_id": self.triggered_ep3_event_id
            }

        ai_year = timeline.get("ai_year")
        agi_year = timeline.get("agi_year")
        asi_year = timeline.get("asi_year")

        if not hasattr(self, '_fired_events_history'):
            self._fired_events_history = set() # Store IDs of all events fired in this run

        candidate_events = []
        for event in self.events:
            if event.id in self._fired_events_history:
                continue

            event_base_year = 0
            if event.stage == "Pre-AGI":
                event_base_year = 0
            elif event.stage == "AGI-Rollout" and agi_year is not None:
                event_base_year = agi_year
            elif event.stage == "Self-Improving" and agi_year is not None:
                event_base_year = agi_year
            elif event.stage == "Post-ASI" and asi_year is not None:
                event_base_year = asi_year

            if current_year == event_base_year + event.year_offset:
                conditions_met = True
                if "ALWAYS" not in event.condition:
                    for cond_id in event.condition:
                        if cond_id not in self._fired_events_history:
                            conditions_met = False
                            break
                if conditions_met:
                    candidate_events.append(event)
        
        self.rng.shuffle(candidate_events)

        newly_fired_events_this_year = []
        for event in candidate_events:
            if event.base_prob > 0 and self.rng.random() <= event.base_prob:
                newly_fired_events_this_year.append(event)
                self._fired_events_history.add(event.id)

                # Retrieve the stochastic parameters for the fired event
                drift_mu_dict, drift_sigma_dict, vol_mu_dict, vol_sigma_dict = \
                    self.event_stats_map.get(event.id, ({}, {}, {}, {}))

                # Apply Stochastic Drift
                for idx, ticker in enumerate(self.tickers):
                    if ticker in drift_mu_dict and ticker in drift_sigma_dict:
                        # Draw a random drift value for this asset for this event
                        stochastic_drift = self.rng.normal(
                            loc=drift_mu_dict[ticker], # Assuming values in JSON are already decimals
                            scale=drift_sigma_dict[ticker]
                        )
                        year_drift[idx] += stochastic_drift
                
                # Apply Asset-Specific Stochastic Volatility Multiplier
                for idx, ticker in enumerate(self.tickers):
                    if ticker in vol_mu_dict and ticker in vol_sigma_dict:
                        # Draw a random volatility multiplier for this asset for this event
                        stochastic_vol_mult = self.rng.normal(
                            loc=vol_mu_dict[ticker],
                            scale=vol_sigma_dict[ticker]
                        )
                        year_vol_multiplier_array[idx] *= stochastic_vol_mult
                
                if event.stage == "Post-ASI":
                    self.triggered_ep3_event_id = event.id

        # final_vol_multiplier = 1.0 + year_vol_change # This line is removed

        return {
            "fired": newly_fired_events_this_year,
            "drift": year_drift,
            "volmul": year_vol_multiplier_array, # This is now an array, not a scalar
            "triggered_ep3_event_id": self.triggered_ep3_event_id
        }

# ---------------------------------- #
# if run as a script for testing (remains largely the same, but note the changes needed in caller)
# ---------------------------------- #
if __name__ == "__main__":
    from engine.timeline_sampler import TimelineSampler

    # Example tickers (replace with actual list as needed)
    tickers = ["PACW","SEMI","VAGS","EMIM","XAIX","ITPS",
               "IWFQ","RBTX","WLDS","MINV","IWFV","EQQQ",
               "MEUD","PRIJ","XCX5","WDEP","LCUK","WCOM",
               "ISPY","NUCG","SGLN"]

    ts = TimelineSampler(
        Path(__file__).parents[1] / "data" / "timeline_buckets.json"
    )
    tl = ts.sample_timeline()
    print("Sampled timeline:", tl)

    horizon_years = 10
    event_engine = EventTreeEngine(
        Path(__file__).parents[1] / "data" / "events_catalogue.json",
        tickers,
        horizon_years=horizon_years
    )

    print("\n--- Simulating Events Year by Year ---")
    current_ep3_event = None
    agi_breakthrough_year = tl.get("agi_year")

    for year in range(horizon_years):
        print(f"\nYear {year}:")
        
        event_results = event_engine.simulate_events(
            timeline=tl,
            current_year=year
        )
        
        newly_fired = event_results["fired"]
        year_drift = event_results["drift"]
        year_volmul = event_results["volmul"] # This will now be an array
        
        current_ep3_event = event_results["triggered_ep3_event_id"]

        if newly_fired:
            print(f"  Fired Events: {[e.id for e in newly_fired]}")
        else:
            print("  No new events fired this year.")
        
        print(f"  Aggregate Drift this year: {year_drift}")
        print(f"  Vol Multiplier this year: {year_volmul}") # This will now be an array
        print(f"  Current E-P3 Event ID (for macro state): {current_ep3_event}")
        
        # Example of how macro state would be determined (conceptual, requires macro.py)
        from . import macro # Assuming macro.py is in the same engine directory
        current_macro_state = macro.get_current_macro_state(
            current_year=year,
            agi_breakthrough_year=agi_breakthrough_year,
            triggered_ep3_event_id=current_ep3_event
        )
        print(f"  Determined Macro State: {current_macro_state}")