# Crop Advisor — real-time crop recommendation & yield estimator

For a plot of land, the system continuously combines **field-sensor soil readings** with **live weather**
and answers two questions:

1. **Which crop should I grow?** (multi-class, top-N with probabilities)
2. **What yield can I expect?** (kg/ha with an 80 % range, and tonnes for the plot's area)

plus a plain-language explanation, agronomic hints and irrigation alerts.
Everything here is software only: you plug your sensors in later by sending HTTP or MQTT messages.

```
 soil sensors / gateway ──HTTP POST──┐
 (ESP32, RPi, LoRa, simulator) ─MQTT─┤
                                     ▼
                         ┌────────────────────────┐   Open-Meteo (live weather + forecast)
 Streamlit dashboard ◄───┤  FastAPI backend       │◄── SoilGrids (pH fallback when no sensor)
 (auto-refresh, what-if) │  SQLite · ML models    │
                         └────────────────────────┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

bash run_backend.sh        # terminal 1  → http://localhost:8000/docs   (first start trains the models, ~40 s)
bash run_frontend.sh        # terminal 2  → http://localhost:8501
python simulator/sensor_simulator.py --plot demo-farm --interval 5   # terminal 3 (optional fake sensor)
```
In the dashboard: sidebar → **Add a plot** (search a place or enter lat/lon; call it *Demo Farm* to match the
simulator command) → watch the **Live** tab update every few seconds. Or `docker compose up` after copying
`.env.example` to `.env`.

No internet? Start the backend with `CA_MOCK_WEATHER=1` to use fake weather (the UI labels it **MOCK**).

## Connecting real sensors

Send whatever your hardware measures; every field is optional (at least one is required):

```bash
curl -X POST http://localhost:8000/api/sensors/readings \
  -H "X-API-Key: dev-sensor-key" -H "Content-Type: application/json" \
  -d '{"plot_id":"demo-farm","device_id":"node-1","n":80,"p":48,"k":40,"ph":6.4,"soil_moisture":31.5,"soil_temp":24.1}'
```
| field | meaning | used by |
|---|---|---|
| `n`, `p`, `k`, `ph` | soil nutrients / acidity | crop + yield models |
| `soil_moisture` (%), `soil_temp` (°C), `ec` | root-zone state | irrigation / stress alerts, charts |
| `timestamp` (ISO-8601) | when measured; server time if omitted | use it when a gateway uploads buffered data via `/api/sensors/readings/batch` |

**MQTT:** set `CA_MQTT_HOST` and publish JSON to `cropadvisor/plots/<plot_id>/readings`.

**Important — units.** The training data expresses N/P/K roughly in kg/ha, while cheap NPK probes report mg/kg.
Calibrate with `CA_N_FACTOR`, `CA_P_FACTOR`, `CA_K_FACTOR` (multipliers applied to raw readings). Validate against a
lab test of the same soil before trusting absolute values.

**Robust to real-world sensor problems**
* readings older than `CA_SENSOR_STALE_MINUTES` (default 60) are treated as *offline* and not used;
* the median of the last 5 readings is used, so single spikes are ignored;
* any missing input (dead probe, API outage) is allowed: the models were trained with random missing-value
  augmentation, and the response includes a `confidence` level and notes explaining what is missing.

## What feeds the model

| model input | live source | definition |
|---|---|---|
| N, P, K, pH | field sensor (fallback for pH: SoilGrids map) | median of recent readings |
| temperature | Open-Meteo | mean of the last 7 completed days, °C |
| humidity | Open-Meteo | mean of the last 7 completed days, % |
| rainfall | Open-Meteo | total over the last 30 days, mm |

Priority: manual what-if override > sensor > live weather > soil map > missing. Every response reports the
`sources` of each input, and the dashboard shows it.

## Data and models

* **Crop classifier** – Random Forest (handles missing values natively; trained with missing-feature augmentation).
* **Yield regressor** – Gradient boosting on log-yield with the crop as a categorical feature; separate quantile
  models give the 80 % range, which is calibrated on held-out data.
* `python -m cropadvisor.ml.train` (run from `backend/`) writes `models/` and prints metrics;
  `profile.json` holds the evaluation used by the **Model insights** tab.

**⚠ By default the models train on SYNTHETIC data** generated from published per-crop requirement profiles, because
the real datasets are not bundled. Accuracy figures on that data (~97 %) are optimistic and the yield numbers are
illustrative only; the dashboard shows a "demo data model" badge until you add real data:

```
data/Crop_recommendation.csv   Kaggle "Crop Recommendation Dataset"  (N,P,K,temperature,humidity,ph,rainfall,label)
data/crop_yield.csv            your yield table: crop,N,P,K,temperature,humidity,ph,rainfall,yield_kg_ha[,region]
```
then run the train command, or `curl -X POST localhost:8000/api/model/retrain -H "X-API-Key: …"`.
Build `crop_yield.csv` from district/state yield data (data.gov.in, ICRISAT, FAO) joined with the weather and soil
averages of the same area/season. If it has a `region`/state column, the leave-one-region-out evaluation uses it.

### Academic angle — built-in experiments (Model insights tab / `profile.json`)
* **Soil vs weather**: accuracy with only soil, only weather, or both (crop choice), and R² for yield with each group removed.
* **Robustness to missing data**: accuracy as 0–6 of 7 inputs disappear, augmented model vs plain model.
* **Regional generalisation**: yield trained on all regions but one, tested on the held-out region.
* Feature importance + per-prediction "influence" (how much a crop's probability drops if one input is unknown).

## API (full docs at `/docs`)

| method | path | |
|---|---|---|
| POST/GET/DELETE | `/api/plots`, `/api/plots/{id}` | manage plots (writes need `X-API-Key`) |
| POST | `/api/sensors/readings`, `/api/sensors/readings/batch` | sensor ingest |
| GET | `/api/plots/{id}/recommendation` | **live** crop + yield + explanation + alerts |
| POST | `/api/plots/{id}/recommendation` | what-if with manual overrides |
| GET | `/api/plots/{id}/readings` · `/sensor` · `/weather` · `/history` | data for charts |
| POST | `/api/predict` | stateless prediction (manual values and/or lat/lon) |
| GET/POST | `/api/model/info`, `/api/model/retrain` | metrics / retrain from `data/` |

## Deploy (Render, free)
1. Push this folder to a GitHub repository (the `.gitignore` already excludes `.env` and the database).
2. On render.com: **New → Blueprint**, pick the repo; Render reads `render.yaml` (Docker, free plan, random `CA_API_KEY`).
3. Wait for the build (models are trained inside the image). Open the URL Render gives you.

The Dockerfile sets `CA_MODE=embedded`: the dashboard runs the backend code inside its own process (see
`LocalClient` in `frontend/api_client.py`), so there is a single Python process and no separate API server. That keeps
memory at about 300 MB, which fits the 512 MB free plan (measured with 4 simultaneous viewers).
It also sets `CA_DEMO_MODE=1`: at startup a `demo-farm` plot with 90 minutes of history is created and a simulated sensor
adds a reading every 10 s (`backend/cropadvisor/demo.py`), because free hosting wipes the disk on every restart.
Set `CA_DEMO_MODE=0` to turn it off. In this hosted mode the sensor HTTP/MQTT API is **not** exposed; for real sensors run
the normal two-process version (`run_backend.sh` + `run_frontend.sh`, or `docker compose up`) with a persistent disk or Postgres.

## Tests
`pytest` (from the project root) — API flow, stale sensors, weather outage, MQTT parsing, model robustness.
Weather/soil calls are mocked there.

## Before going to production
* Set a long random `CA_API_KEY` (and restrict `CA_CORS_ORIGINS`). Read endpoints are open by default; put the API behind
  a reverse proxy with HTTPS and user authentication if plots are private.
* SQLite is fine for a pilot; move readings to Postgres/TimescaleDB for many devices (`db.py` is the only data layer).
* Recommendations are decision support, not agronomic prescriptions; confirm fertiliser doses locally.

## Layout
```
backend/cropadvisor/   main.py (API) · db.py · config.py · schemas.py · mqtt_listener.py
                       services/{weather,soil,pipeline}.py · ml/{data,profiles,train,predictor}.py
frontend/              app.py (Streamlit) · api_client.py
simulator/             sensor_simulator.py        tests/    pytest suite
```
