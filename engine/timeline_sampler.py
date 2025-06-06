# AI_Scenario_Sim/engine/timeline_sampler.py
# ------------------------------------------
"""
TimelineSampler
===============

Draws a 3-point capability timeline — Narrow-AI (ai), AGI (agi), ASI (asi) —
according to the probability buckets in data/timeline_buckets.json
and provides a helper to map absolute year → lifecycle stage.

Stages returned by `stage_for_year`:
    "Narrow-AI", "Pre-AGI", "AGI-Rollout", "Self-Improving", "Post-ASI"
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json, numpy as np

# ---------- Data structures ---------------------------------------------------

@dataclass
class Bucket:
    name: str
    prob: float
    agi_min: int
    agi_max: int
    asi_lag: int        # yrs after AGI that ASI emerges

# ---------- Sampler class -----------------------------------------------------

class TimelineSampler:
    def __init__(self, cfg_path: str | Path, seed: int | None = None):
        cfg_path = Path(cfg_path)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        self.ai_offset = int(cfg["ai_offset"])
        self.buckets: List[Bucket] = [Bucket(**b) for b in cfg["buckets"]]
        probs = [b.prob for b in self.buckets]

        if abs(sum(probs) - 1) > 1e-6:
            raise ValueError("Bucket probabilities must sum to 1.")

        self._bucket_probs = np.array(probs, dtype=float)
        # seed=None → nondeterministic RNG
        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    def sample_timeline(self) -> Dict[str, int | str]:
        """
        Returns a dict:
            {'ai': years_ahead,
             'agi': years_ahead,
             'asi': years_ahead (10_000 if never),
             'bucket': 'early'|'middle'|'late'|'never'}
        """
        bucket: Bucket = self.rng.choice(self.buckets, p=self._bucket_probs)
        agi = int(self.rng.integers(bucket.agi_min, bucket.agi_max + 1))
        ai  = agi - self.ai_offset
        asi = agi + bucket.asi_lag if bucket.name != "never" else 10_000
        return {"ai": ai, "agi": agi, "asi": asi, "bucket": bucket.name}

    # -------------------------------------------------------------------------
    @staticmethod
    def _stage_labels():  # internal helper
        return ("Narrow-AI", "Pre-AGI", "AGI-Rollout",
                "Self-Improving", "Post-ASI")

    # -------------------------------------------------------------------------
    def stage_for_year(self, tl: Dict[str, int], y: int) -> str:
        """
        Given a timeline dict from sample_timeline() and an absolute year offset,
        return which lifecycle stage that year belongs to.
        """
        if y < tl["ai"]:
            return "Narrow-AI"
        if y < tl["agi"]:
            return "Pre-AGI"
        if y < tl["asi"]:
            return "AGI-Rollout"
        if y < tl["asi"] + 15:          # first 15 yr of recursive ASI
            return "Self-Improving"
        return "Post-ASI"

# -----------------------------------------------------------------------------#
# Stand-alone test
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    ts = TimelineSampler(Path(__file__).parents[1] / "data" / "timeline_buckets.json")
    for _ in range(3):
        tl = ts.sample_timeline()
        print("Sampled:", tl)
        print("  Stage in year 12 →", ts.stage_for_year(tl, 12))
