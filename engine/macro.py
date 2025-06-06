"""Utilities for loading macro-state parameters used in simulations."""
from __future__ import annotations
import json
from pathlib import Path
import random # Import random for probability sampling

# Path to the data directory two levels up from this file
BASE = Path(__file__).resolve().parent.parent / "data"
_macro_path = BASE / "macro_states.json"

with open(_macro_path, "r") as f:
    _raw_macros = json.load(f)

# Public constants for state names and their base probabilities (these remain as lists)
MACRO_STATES = [entry["state"] for entry in _raw_macros]
MACRO_PROBS = [entry["prob"] for entry in _raw_macros] # These are BASE probabilities, potentially overridden

# --- CORRECTED MACRO_MAP CREATION ---
# MACRO_MAP now stores dictionaries for mu_shift and sigma_mult per state.
# We process _raw_macros to correctly store the asset-specific shifts and multipliers.
MACRO_MAP = {}
for entry in _raw_macros:
    # Convert percentages in mu_shift dictionary to decimals for each asset
    mu_shifts_decimal_dict = {
        ticker: val / 100.0 for ticker, val in entry["mu_shift"].items()
    }
    # sigma_mults are already multipliers, so no division needed
    sigma_mults_dict = entry["sigma_mult"]

    MACRO_MAP[entry["state"]] = (mu_shifts_decimal_dict, sigma_mults_dict)

# --- NEW FUNCTION FOR DYNAMIC MACRO STATE SELECTION ---

def get_current_macro_state(
    current_year: int,
    agi_breakthrough_year: int | None,
    triggered_ep3_event_id: str | None
) -> str:
    """
    Determines the current macro state dynamically based on AGI events.

    Args:
        current_year: The current year of the simulation.
        agi_breakthrough_year: The year AGI breakthrough occurred (if any),
                               or None if no breakthrough has occurred yet.
        triggered_ep3_event_id: The ID of the Post-ASI (E-P3) event that has been triggered,
                                  or None if no such event has triggered yet.

    Returns:
        The selected macro state for the current year.
    """
    ep3_to_macro_state_map = {
        "E-P3-Abundance": "Abundance",
        "E-P3-Oligopoly": "Oligopoly",
        "E-P3-Managed": "Managed",
        "E-P3-Conflict": "Conflict",
        "E-P3-EquitableDistribution": "Managed" # Assuming EquitableDistribution leads to a 'Managed' economic state
    }

    # If a specific E-P3 event has been triggered, its corresponding macro state dominates.
    if triggered_ep3_event_id and triggered_ep3_event_id in ep3_to_macro_state_map:
        return ep3_to_macro_state_map[triggered_ep3_event_id]
    else:
        # If no E-P3 event dictates the macro state, sample from base probabilities
        # adjusted for whether AGI breakthrough has occurred.
        
        # Prepare states and probabilities for sampling
        current_available_states = []
        current_available_probs = []

        if agi_breakthrough_year is None or current_year < agi_breakthrough_year:
            # Before AGI breakthrough, all base macro states are considered
            current_available_states = MACRO_STATES
            current_available_probs = MACRO_PROBS
        else:
            # After AGI breakthrough, 'NoBreakthrough' state is typically excluded
            for i, state in enumerate(MACRO_STATES):
                if state == "NoBreakthrough":
                    continue # Skip 'NoBreakthrough' if AGI has occurred
                current_available_states.append(state)
                current_available_probs.append(MACRO_PROBS[i])

        # --- Handle edge cases before attempting to normalize and sample ---
        if not current_available_states:
            # This means no valid macro states are left to choose from
            print("Error: No available macro states to choose from after filtering. Defaulting to NoBreakthrough.")
            return "NoBreakthrough"
        
        total_prob = sum(current_available_probs)
        if total_prob <= 0:
            # If probabilities sum to zero or negative, cannot sample probabilistically
            print("Warning: Sum of probabilities for available macro states is zero or negative. Defaulting to NoBreakthrough.")
            return "NoBreakthrough"

        # Normalize probabilities for sampling
        normalized_probs = [p / total_prob for p in current_available_probs]

        # Select a macro state based on the normalized probabilities
        return random.choices(current_available_states, weights=normalized_probs, k=1)[0]