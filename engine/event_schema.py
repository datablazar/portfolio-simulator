from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Event:
    id:            str
    stage:         str
    year_offset:   int
    base_prob:     float
    condition:     List[str]
    delta_drift:   Dict[str, float] = field(default_factory=dict)
    delta_vol:     float = 0.0
    next_events:   List[Tuple[str, float]] = field(default_factory=list)
    narrative:     str = ""

    def drift_vector(self, tickers: List[str]) -> List[float]:
        return [self.delta_drift.get(t, 0.0) for t in tickers]
