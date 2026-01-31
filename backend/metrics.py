from dataclasses import dataclass, field
from typing import List
from config import IDLE_EMISSION_G_PER_SEC, MOVING_EMISSION_G_PER_SEC

@dataclass
class Metrics:
    total_wait_time: float = 0.0
    vehicles_served: int = 0
    idle_emissions_g: float = 0.0
    moving_emissions_g: float = 0.0

    # time series for plotting
    t_series: List[int] = field(default_factory=list)
    wait_series: List[float] = field(default_factory=list)
    queue_series: List[int] = field(default_factory=list)
    emissions_series: List[float] = field(default_factory=list)
    served_series: List[int] = field(default_factory=list)

    def record_step(self, t: int, total_queue: int, moved: int, dt: float):
        self.total_wait_time += total_queue * dt
        self.idle_emissions_g += total_queue * IDLE_EMISSION_G_PER_SEC * dt
        self.moving_emissions_g += moved * MOVING_EMISSION_G_PER_SEC * dt
        self.vehicles_served += moved

        # store time series (keep it light)
        self.t_series.append(t)
        self.queue_series.append(total_queue)
        self.wait_series.append(self.avg_wait_per_vehicle())
        self.emissions_series.append(self.total_emissions_g())
        self.served_series.append(self.vehicles_served)

    def total_emissions_g(self) -> float:
        return self.idle_emissions_g + self.moving_emissions_g

    def avg_wait_per_vehicle(self) -> float:
        if self.vehicles_served <= 0:
            return 0.0
        return self.total_wait_time / self.vehicles_served
