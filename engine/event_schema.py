from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Event:
    id:            str
    stage:         str
    year_offset:   int
    base_prob:     float
    condition:     List[str]
    next_events:   List[Tuple[str, float]] = field(default_factory=list)
    narrative:     str = ""