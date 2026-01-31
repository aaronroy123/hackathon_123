# Ulta-WOW Smart City Traffic Optimizer (Hackathon Demo)

This project is a **real-time traffic signal optimizer** demo with:
- 4 intersections (A–D) with **interdependencies**
- **Side-by-side** comparison: **Fixed-time** vs **Dynamic** optimizer (same traffic stream)
- **Emergency vehicle priority** + **Pedestrian crossing minimum**
- **Auto-events** (random emergencies/ped requests) + manual inject buttons
- Live dashboard with **charts** + **city map** + congestion/health score

## Run (Backend)
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Run (Frontend)
Open `frontend/index.html` in your browser.

> If your browser blocks `http://localhost:8000` due to CORS, it's already enabled in the backend.

## Demo Flow (60–90s)
1. Click **Start**
2. Choose **Morning Peak** (queues grow in one direction)
3. Watch **Fixed** build queues while **Dynamic** stabilizes
4. Toggle **Auto Events**, or click **Inject Emergency** and **Pedestrian Request**
5. Click **Run Compare (5 min)** to show improvement numbers

## Notes
- "Real-time" is simulated sensor streaming (queues/arrivals/events), as is standard for hackathons.
- Emissions are an **idle-time proxy** (queued vehicles emit more).

