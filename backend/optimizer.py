from typing import Dict, Tuple
from config import MIN_GREEN, MAX_GREEN, PED_MIN

PHASES = {
    "NS": ["N", "S"],
    "EW": ["E", "W"],
}

def sum_queue(queues: Dict[str, int], phase: str) -> int:
    return sum(queues.get(a, 0) for a in PHASES[phase])

def choose_phase(
    queues: Dict[str, int],
    downstream: Dict[str, int],     # kept for signature compatibility (not used)
    emergency: Dict[str, bool],
    ped: Dict[str, bool],
    current_phase: str = "NS",
) -> Tuple[str, int, str, bool]:
    """
    Returns: (chosen_phase, green_seconds, reason, is_emergency)

    - Emergency override: choose phase containing emergency approach.
    - Else choose larger queue side (NS vs EW) with hysteresis.
    - Green scales with demand, capped.
    - Ped forces minimum green.
    """

    # 1) Emergency override
    for phase, apps in PHASES.items():
        if any(emergency.get(a, False) for a in apps):
            return phase, MAX_GREEN, f"emergency priority on {phase}", True

    # 2) Ped requests per phase
    ped_needed = {ph: any(ped.get(a, False) for a in apps) for ph, apps in PHASES.items()}

    # 3) Queue totals
    q_ns = sum_queue(queues, "NS")
    q_ew = sum_queue(queues, "EW")

    # 4) Winner by queue
    winner = "NS" if q_ns >= q_ew else "EW"
    q_current = q_ns if current_phase == "NS" else q_ew
    q_winner  = q_ns if winner == "NS" else q_ew

    # 5) Hysteresis
    HYST = 3
    if winner == current_phase:
        chosen = current_phase
    else:
        chosen = winner if q_winner >= (q_current + HYST) else current_phase

    # 6) Green scaling
    chosen_q = q_ns if chosen == "NS" else q_ew
    extra = min(18, chosen_q // 2)
    green = MIN_GREEN + extra
    green = max(MIN_GREEN, min(MAX_GREEN, green))

    # 7) Ped minimum
    if ped_needed[chosen]:
        green = max(green, PED_MIN)

    return chosen, green, f"queue NS={q_ns} EW={q_ew} | chosen={chosen} green={green}s", False
