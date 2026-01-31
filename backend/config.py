SIM_DT = 1.0                 # seconds per simulation step
CONTROL_INTERVAL = 5         # seconds between control decisions

MIN_GREEN = 10
MAX_GREEN = 40
YELLOW = 3
ALL_RED = 1

PED_MIN = 8                  # minimum green to safely cross
SERVICE_RATE = 0.65          # vehicles/sec per served approach (per second probability)

# Emissions proxy (grams/sec per vehicle)
IDLE_EMISSION_G_PER_SEC = 2.5
MOVING_EMISSION_G_PER_SEC = 1.2

# Default scenario arrival rates (vehicles/sec) for each intersection approach
# A->B->C->D corridor: heavy eastbound in morning peak, heavy westbound in evening peak
SCENARIOS = {
    "normal": {
        "A": {"N":0.10,"S":0.08,"E":0.22,"W":0.12},
        "B": {"N":0.10,"S":0.10,"E":0.25,"W":0.18},
        "C": {"N":0.08,"S":0.10,"E":0.23,"W":0.20},
        "D": {"N":0.09,"S":0.08,"E":0.18,"W":0.16},
    },
    "morning_peak": {
        "A": {"N":0.12,"S":0.10,"E":0.40,"W":0.08},
        "B": {"N":0.12,"S":0.12,"E":0.42,"W":0.12},
        "C": {"N":0.10,"S":0.12,"E":0.38,"W":0.14},
        "D": {"N":0.10,"S":0.10,"E":0.30,"W":0.16},
    },
    "evening_peak": {
        "A": {"N":0.10,"S":0.10,"E":0.12,"W":0.35},
        "B": {"N":0.10,"S":0.12,"E":0.14,"W":0.42},
        "C": {"N":0.10,"S":0.12,"E":0.16,"W":0.40},
        "D": {"N":0.10,"S":0.10,"E":0.18,"W":0.32},
    },
    "accident_eastbound": {
        # simulates reduced discharge on EW for intersection B (handled in simulator)
        "A": {"N":0.10,"S":0.08,"E":0.30,"W":0.10},
        "B": {"N":0.10,"S":0.10,"E":0.35,"W":0.12},
        "C": {"N":0.08,"S":0.10,"E":0.28,"W":0.18},
        "D": {"N":0.09,"S":0.08,"E":0.22,"W":0.16},
    },
}
