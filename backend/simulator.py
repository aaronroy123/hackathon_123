from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from config import (
    SIM_DT,
    CONTROL_INTERVAL,
    MIN_GREEN,
    MAX_GREEN,
    YELLOW,
    ALL_RED,
    SERVICE_RATE,
    SCENARIOS,
)
from metrics import Metrics
from optimizer import choose_phase

PHASE_SERVES = {"NS": ["N", "S"], "EW": ["E", "W"]}


@dataclass
class LightState:
    phase: str = "NS"
    time_left: int = MIN_GREEN

    in_transition: bool = False
    transition_left: int = 0
    next_phase: Optional[str] = None

    # ✅ track how long current green has been active
    green_elapsed: int = 0


class Intersection:
    def __init__(self, iid: str, arrival_rates: Dict[str, float]):
        self.id = iid
        self.queues: Dict[str, int] = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.arrival_rate = dict(arrival_rates)

        self.emergency: Dict[str, bool] = {a: False for a in self.queues}
        self.ped: Dict[str, bool] = {a: False for a in self.queues}

        self.light = LightState()
        self.last_decision_reason = ""

        self.capacity_mult = 1.0

    def snapshot(self) -> Dict:
        total_q = sum(self.queues.values())
        score = max(0, 100 - int(total_q * 3))
        return {
            "id": self.id,
            "phase": self.light.phase,
            "time_left": self.light.time_left,
            "in_transition": self.light.in_transition,
            "transition_left": self.light.transition_left,
            "next_phase": self.light.next_phase,
            "queues": self.queues,
            "arrival_rate": self.arrival_rate,
            "emergency": self.emergency,
            "ped": self.ped,
            "health_score": score,
            "reason": self.last_decision_reason,
        }

    def set_event(self, kind: str, approach_or_phase: str):
        if kind == "emergency":
            if approach_or_phase in self.emergency:
                self.emergency[approach_or_phase] = True
        elif kind == "ped":
            if approach_or_phase == "NS":
                self.ped["N"] = True
                self.ped["S"] = True
            elif approach_or_phase == "EW":
                self.ped["E"] = True
                self.ped["W"] = True

    def clear_served_ped(self):
        for a in PHASE_SERVES[self.light.phase]:
            self.ped[a] = False

    def arrivals_step(self, rng: np.random.Generator):
        for a, r in self.arrival_rate.items():
            if rng.random() < r * SIM_DT:
                self.queues[a] += 1

    def service_step(self, rng: np.random.Generator) -> Dict[str, int]:
        if self.light.in_transition:
            return {"N": 0, "S": 0, "E": 0, "W": 0}

        moved = {"N": 0, "S": 0, "E": 0, "W": 0}
        served_apps = PHASE_SERVES[self.light.phase]

        for a in served_apps:
            if self.queues[a] > 0:
                p = (SERVICE_RATE * self.capacity_mult) * SIM_DT
                if rng.random() < p:
                    self.queues[a] -= 1
                    moved[a] += 1

                    # emergency clears when that approach queue becomes 0
                    if self.emergency[a] and self.queues[a] == 0:
                        self.emergency[a] = False

        return moved

    def tick_signal(self):
        # transition countdown (yellow+all-red or all-red only)
        if self.light.in_transition:
            self.light.transition_left -= 1
            if self.light.transition_left <= 0:
                self.light.in_transition = False
                self.light.phase = self.light.next_phase or self.light.phase
                self.light.next_phase = None
                self.light.green_elapsed = 0  # ✅ reset when new green starts
            return

        # green running
        if self.light.time_left > 0:
            self.light.time_left -= 1
            self.light.green_elapsed += 1

        # green ended
        if self.light.time_left == 0:
            self.clear_served_ped()

    def apply_control(self, phase: str, green_for: int, reason: str = "", emergency_preempt: bool = False):
        if self.light.in_transition:
            return

        if phase != self.light.phase:
            self.light.in_transition = True
            # emergency preempt = immediate all-red clearance only
            self.light.transition_left = (ALL_RED if emergency_preempt else (YELLOW + ALL_RED))
            self.light.next_phase = phase

            # after transition ends, this becomes the new green time
            self.light.time_left = green_for
        else:
            # if staying in same phase and we are at end, set a new window
            if self.light.time_left == 0:
                self.light.time_left = green_for
                self.light.green_elapsed = 0

        self.last_decision_reason = reason


class CitySim:
    def __init__(self, sim_name: str, mode: str, scenario: str, seed: int, auto_events: bool = True):
        self.sim_name = sim_name
        self.mode = mode
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.metrics = Metrics()
        self.auto_events = auto_events

        rates = SCENARIOS[scenario]
        self.intersections: Dict[str, Intersection] = {
            iid: Intersection(iid, rates[iid]) for iid in ["A", "B", "C", "D"]
        }

        self.fixed_cycle = 60
        self.fixed_ns = 30
        self.fixed_ew = 30

        if scenario == "accident_eastbound":
            self.intersections["B"].capacity_mult = 0.45

    def set_scenario(self, scenario: str):
        self.__init__(
            self.sim_name,
            self.mode,
            scenario,
            seed=int(self.rng.integers(1, 1_000_000)),
            auto_events=self.auto_events,
        )

    def inject_event(self, intersection_id: str, kind: str, where: str):
        if intersection_id in self.intersections:
            self.intersections[intersection_id].set_event(kind, where)

    def maybe_auto_events(self):
        if not self.auto_events:
            return

        if self.t > 0 and self.t % int(self.rng.integers(80, 140)) == 0:
            iid = self.rng.choice(["A", "B", "C", "D"])
            phase = self.rng.choice(["NS", "EW"])
            self.inject_event(iid, "ped", phase)

        if self.t > 0 and self.t % int(self.rng.integers(120, 200)) == 0:
            iid = self.rng.choice(["A", "B", "C", "D"])
            approach = self.rng.choice(["N", "S", "E", "W"])
            self.inject_event(iid, "emergency", approach)

    def downstream_queues(self, iid: str) -> Dict[str, int]:
        down = {"N": 0, "S": 0, "E": 0, "W": 0}
        if iid == "A":
            down["E"] = self.intersections["B"].queues["W"]
        if iid == "B":
            down["E"] = self.intersections["C"].queues["W"]
            down["W"] = self.intersections["A"].queues["E"]
        if iid == "C":
            down["E"] = self.intersections["D"].queues["W"]
            down["W"] = self.intersections["B"].queues["E"]
        if iid == "D":
            down["W"] = self.intersections["C"].queues["E"]
        return down

    def controller_tick_fixed(self):
        offsets = {"A": 0, "B": 10, "C": 20, "D": 30}
        for iid, inter in self.intersections.items():
            if inter.light.in_transition:
                continue
            if inter.light.time_left != 0:
                continue

            cycle_pos = (self.t + offsets.get(iid, 0)) % self.fixed_cycle
            desired = "NS" if cycle_pos < self.fixed_ns else "EW"
            inter.apply_control(desired, green_for=MIN_GREEN, reason="fixed schedule")

    def controller_tick_dynamic(self):
        """
        ✅ Key behavior:
        - If emergency exists: immediately switch to emergency axis (with all-red clearance)
        - Else: choose the axis with bigger queue (NO hysteresis)
        - Allow switching mid-green when other axis becomes bigger,
          BUT only after MIN_GREEN seconds already served (prevents flicker)
        """
        for iid, inter in self.intersections.items():
            if inter.light.in_transition:
                continue

            down = self.downstream_queues(iid)

            # always compute "best now"
            phase, green, reason, is_emergency = choose_phase(
                inter.queues,
                down,
                inter.emergency,
                inter.ped,
                current_phase=inter.light.phase,
            )

            # Emergency can preempt anytime
            if is_emergency:
                inter.apply_control(
                    phase,
                    green_for=green,
                    reason=reason,
                    emergency_preempt=True,
                )
                continue

            # Normal: if green ended, pick next window
            if inter.light.time_left == 0:
                inter.apply_control(phase, green_for=green, reason=reason)
                continue

            # Normal: mid-green switching allowed if other axis becomes bigger
            if phase != inter.light.phase and inter.light.green_elapsed >= MIN_GREEN:
                inter.apply_control(phase, green_for=green, reason="switch due to longer queue")
                continue

            # else keep current green (do nothing)

    def step(self) -> Dict:
        self.t += 1
        self.maybe_auto_events()

        for inter in self.intersections.values():
            inter.arrivals_step(self.rng)

        moved_total = 0
        corridor_push = {("A", "B"): 0, ("B", "C"): 0, ("C", "D"): 0}

        for iid, inter in self.intersections.items():
            moved_by_dir = inter.service_step(self.rng)
            moved_total += sum(moved_by_dir.values())

            if (not inter.light.in_transition) and inter.light.phase == "EW":
                if iid in ["A", "B", "C"]:
                    eastbound_served = moved_by_dir["W"]
                    corridor_push[(iid, chr(ord(iid) + 1))] = int(max(0, eastbound_served * 0.90))

        for (src, dst), count in corridor_push.items():
            if count > 0 and dst in self.intersections:
                self.intersections[dst].queues["W"] += count

        for inter in self.intersections.values():
            inter.tick_signal()

        if self.t % CONTROL_INTERVAL == 0:
            if self.mode == "fixed":
                self.controller_tick_fixed()
            else:
                self.controller_tick_dynamic()

        total_queue = sum(sum(inter.queues.values()) for inter in self.intersections.values())
        self.metrics.record_step(t=self.t, total_queue=total_queue, moved=moved_total, dt=SIM_DT)

        return self.snapshot()

    def snapshot(self) -> Dict:
        return {
            "name": self.sim_name,
            "mode": self.mode,
            "scenario": self.scenario,
            "t": self.t,
            "metrics": {
                "vehicles_served": self.metrics.vehicles_served,
                "avg_wait": round(self.metrics.avg_wait_per_vehicle(), 3),
                "emissions_g": round(self.metrics.total_emissions_g(), 1),
                "total_queue": sum(sum(i.queues.values()) for i in self.intersections.values()),
                "series": {
                    "t": self.metrics.t_series[-300:],
                    "avg_wait": self.metrics.wait_series[-300:],
                    "queue": self.metrics.queue_series[-300:],
                    "emissions": self.metrics.emissions_series[-300:],
                },
            },
            "intersections": [inter.snapshot() for inter in self.intersections.values()],
        }


class TwinCity:
    def __init__(self, scenario: str = "normal", seed: int = 42):
        self.seed = seed
        self.scenario = scenario
        self.fixed = CitySim("fixed", "fixed", scenario, seed=seed, auto_events=True)
        self.dynamic = CitySim("dynamic", "dynamic", scenario, seed=seed, auto_events=True)
        self.running = False

    def set_running(self, running: bool):
        self.running = running

    def set_auto_events(self, enabled: bool):
        self.fixed.auto_events = enabled
        self.dynamic.auto_events = enabled

    def set_scenario(self, scenario: str):
        self.scenario = scenario
        self.fixed = CitySim("fixed", "fixed", scenario, seed=self.seed, auto_events=self.fixed.auto_events)
        self.dynamic = CitySim("dynamic", "dynamic", scenario, seed=self.seed, auto_events=self.dynamic.auto_events)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        self.set_scenario(self.scenario)

    def inject_event_both(self, intersection_id: str, kind: str, where: str):
        self.fixed.inject_event(intersection_id, kind, where)
        self.dynamic.inject_event(intersection_id, kind, where)

    def tick(self) -> Dict:
        if not self.running:
            return {"running": False, "fixed": self.fixed.snapshot(), "dynamic": self.dynamic.snapshot()}
        return {"running": True, "fixed": self.fixed.step(), "dynamic": self.dynamic.step()}

    def run_compare(self, steps: int = 300) -> Dict:
        base_seed = self.seed
        f = CitySim("fixed", "fixed", self.scenario, seed=base_seed, auto_events=False)
        d = CitySim("dynamic", "dynamic", self.scenario, seed=base_seed, auto_events=False)

        scripted = [
            (80, "ped", "C", "NS"),
            (140, "emergency", "B", "E"),
            (210, "ped", "A", "EW"),
        ]

        for t in range(1, steps + 1):
            for when, kind, iid, where in scripted:
                if t == when:
                    f.inject_event(iid, kind, where)
                    d.inject_event(iid, kind, where)
            f.step()
            d.step()

        fm = f.metrics
        dm = d.metrics

        def pct_improve(baseline: float, improved: float) -> float:
            if baseline == 0:
                return 0.0
            return round((baseline - improved) / baseline * 100.0, 2)

        return {
            "steps": steps,
            "scenario": self.scenario,
            "fixed": {
                "vehicles_served": fm.vehicles_served,
                "avg_wait": round(fm.avg_wait_per_vehicle(), 3),
                "emissions_g": round(fm.total_emissions_g(), 1),
            },
            "dynamic": {
                "vehicles_served": dm.vehicles_served,
                "avg_wait": round(dm.avg_wait_per_vehicle(), 3),
                "emissions_g": round(dm.total_emissions_g(), 1),
            },
            "improvement": {
                "avg_wait_reduction_pct": pct_improve(fm.avg_wait_per_vehicle(), dm.avg_wait_per_vehicle()),
                "emissions_reduction_pct": pct_improve(fm.total_emissions_g(), dm.total_emissions_g()),
                "throughput_increase_pct": round(
                    ((dm.vehicles_served - fm.vehicles_served) / (fm.vehicles_served or 1)) * 100.0, 2
                ),
            },
        }
