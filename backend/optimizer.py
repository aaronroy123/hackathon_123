from typing import Dict, Tuple
from config import MIN_GREEN, MAX_GREEN, PED_MIN

PHASES = {
    "NS": ["N", "S"],
    "EW": ["E", "W"],
}

def compute_pressure(queues: Dict[str, int], downstream: Dict[str, int], phase: str) -> int:
    served = PHASES[phase]
    up = sum(queues.get(a, 0) for a in served)
    down = sum(downstream.get(a, 0) for a in served)
    return up - down

def choose_phase(queues: Dict[str, int],
                 downstream: Dict[str, int],
                 emergency: Dict[str, bool],
                 ped: Dict[str, bool]) -> Tuple[str, int, str]:
    # Emergency override
    for phase, apps in PHASES.items():
        if any(emergency.get(a, False) for a in apps):
            return phase, MAX_GREEN, "emergency priority"

    ped_needed = {phase: any(ped.get(a, False) for a in apps) for phase, apps in PHASES.items()}

    p_ns = compute_pressure(queues, downstream, "NS")
    p_ew = compute_pressure(queues, downstream, "EW")

    best = "NS" if p_ns >= p_ew else "EW"
    diff = abs(p_ns - p_ew)

    green = MIN_GREEN + min(20, diff)
    green = max(MIN_GREEN, min(MAX_GREEN, green))

    if ped_needed[best]:
        green = max(green, PED_MIN)

    return best, green, f"pressure NS={p_ns}, EW={p_ew}"
