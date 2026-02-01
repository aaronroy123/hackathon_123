from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from simulator import TwinCity

app = FastAPI(title="Ulta-WOW Traffic Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

city = TwinCity(scenario="normal", seed=42)

class ControlRequest(BaseModel):
    running: Optional[bool] = None
    scenario: Optional[str] = None
    auto_events: Optional[bool] = None
    seed: Optional[int] = None

class InjectRequest(BaseModel):
    intersection_id: str
    kind: str   # "emergency" | "ped"
    where: str  # emergency: N/S/E/W ; ped: NS/EW

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/control")
def control(req: ControlRequest):
    if req.seed is not None:
        city.reset(seed=req.seed)
    if req.scenario is not None:
        city.set_scenario(req.scenario)
    if req.auto_events is not None:
        city.set_auto_events(req.auto_events)
    if req.running is not None:
        city.set_running(req.running)

    return {
        "running": city.running,
        "scenario": city.scenario,
        "seed": city.seed,
        "auto_events": city.fixed.auto_events,
    }

@app.post("/inject")
def inject(req: InjectRequest):
    city.inject_event_both(req.intersection_id, req.kind, req.where)
    return {"ok": True}

@app.get("/state")
def state():
    return {
        "running": city.running,
        "fixed": city.fixed.snapshot(),
        "dynamic": city.dynamic.snapshot()
    }

@app.post("/tick")
def tick():
    return city.tick()

@app.get("/compare")
def compare(steps: int = 300):
    return city.run_compare(steps=steps)
