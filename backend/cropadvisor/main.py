"""FastAPI application.

    cd backend && uvicorn cropadvisor.main:app --host 0.0.0.0 --port 8000
    open http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import secrets
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import db, demo, mqtt_listener
from .config import get_settings
from .ml.predictor import get_predictor, reload_predictor
from .schemas import AdhocRequest, PlotIn, SensorReading
from .services import pipeline, soil, weather

log = logging.getLogger("cropadvisor")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    s = get_settings()
    # load (or train on first run) models in the background so the API is reachable immediately
    threading.Thread(target=get_predictor, args=(s.models_dir,), daemon=True).start()
    app.state.mqtt = mqtt_listener.start()
    app.state.demo = demo.start()
    yield
    if app.state.demo:
        app.state.demo.set()
    if app.state.mqtt:
        app.state.mqtt.loop_stop()


app = FastAPI(title="Crop Advisor API", version="1.0.0", lifespan=lifespan,
              description="Real-time crop recommendation and yield estimation from field sensors + live weather.")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins.split(","), allow_methods=["*"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(None)):
    if not x_api_key or not secrets.compare_digest(x_api_key, get_settings().api_key):
        raise HTTPException(401, "missing or invalid X-API-Key header")


def _plot_or_404(plot_id: str) -> dict:
    plot = db.get_plot(plot_id)
    if not plot:
        raise HTTPException(404, f"plot '{plot_id}' not found")
    return plot


# ----------------------------------------------------------------------- meta
@app.get("/health")
def health():
    s = get_settings()
    ready = (s.models_dir / "crop_classifier.joblib").exists()
    return {"status": "ok", "models_ready": ready, "mqtt": bool(s.mqtt_host)}


@app.get("/api/model/info")
def model_info():
    try:
        return get_predictor(get_settings().models_dir).info()
    except Exception as exc:
        raise HTTPException(503, f"models not ready: {exc}")


@app.post("/api/model/retrain", dependencies=[Depends(require_key)])
def retrain():
    """Re-train from whatever is in data/ (real CSVs if present) and hot-reload."""
    from .ml import train

    train.main()
    return reload_predictor(get_settings().models_dir).info()["metrics"]["classifier"]["complete_data"]


# ----------------------------------------------------------------------- plots
@app.post("/api/plots", status_code=201, dependencies=[Depends(require_key)])
def create_plot(body: PlotIn):
    try:
        return db.create_plot(body.id, body.name, body.lat, body.lon, body.area_ha)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/plots")
def list_plots():
    return db.list_plots()


@app.get("/api/plots/{plot_id}")
def get_plot(plot_id: str):
    return _plot_or_404(plot_id)


@app.delete("/api/plots/{plot_id}", status_code=204, dependencies=[Depends(require_key)])
def delete_plot(plot_id: str):
    if not db.delete_plot(plot_id):
        raise HTTPException(404, "not found")


# ----------------------------------------------------------------------- sensors (real-time ingest)
@app.post("/api/sensors/readings", status_code=201, dependencies=[Depends(require_key)])
def ingest(reading: SensorReading):
    _plot_or_404(reading.plot_id)
    rid = db.insert_reading(reading.plot_id, reading.device_id, reading.timestamp or db.utcnow(), reading.model_dump())
    return {"id": rid, "plot_id": reading.plot_id}


@app.post("/api/sensors/readings/batch", status_code=201, dependencies=[Depends(require_key)])
def ingest_batch(readings: list[SensorReading]):
    """For gateways that buffer while offline and flush later (send the original timestamps)."""
    if len(readings) > 1000:
        raise HTTPException(413, "max 1000 readings per batch")
    for r in readings:
        _plot_or_404(r.plot_id)
    for r in readings:
        db.insert_reading(r.plot_id, r.device_id, r.timestamp or db.utcnow(), r.model_dump())
    return {"stored": len(readings)}


@app.get("/api/plots/{plot_id}/readings")
def readings(plot_id: str, limit: int = Query(200, ge=1, le=5000)):
    _plot_or_404(plot_id)
    return list(reversed(db.list_readings(plot_id, limit=limit)))  # oldest -> newest, ready to chart


@app.get("/api/plots/{plot_id}/sensor")
def sensor_status(plot_id: str):
    _plot_or_404(plot_id)
    return pipeline.sensor_snapshot(plot_id)


# ----------------------------------------------------------------------- weather / soil
@app.get("/api/plots/{plot_id}/weather")
def plot_weather(plot_id: str):
    plot = _plot_or_404(plot_id)
    try:
        return weather.fetch_weather(plot["lat"], plot["lon"])
    except weather.WeatherUnavailable as exc:
        raise HTTPException(503, f"weather service unavailable: {exc}")


@app.get("/api/plots/{plot_id}/soil-map")
def plot_soil_map(plot_id: str):
    plot = _plot_or_404(plot_id)
    return soil.fetch_soil_ph(plot["lat"], plot["lon"]) or {}


# ----------------------------------------------------------------------- recommendations
@app.get("/api/plots/{plot_id}/recommendation")
def recommendation(plot_id: str, top_k: int = Query(3, ge=1, le=10), crop: str | None = None, log: bool = True):
    """Live recommendation: latest sensor data + live weather + models."""
    plot = _plot_or_404(plot_id)
    try:
        return pipeline.build_recommendation(plot, crop=crop, top_k=top_k, log=log)
    except FileNotFoundError:
        raise HTTPException(503, "models are still training, retry in a minute")


@app.post("/api/plots/{plot_id}/recommendation")
def recommendation_whatif(plot_id: str, body: AdhocRequest):
    """What-if: same as GET but with manual overrides on top of the live data."""
    plot = _plot_or_404(plot_id)
    return pipeline.build_recommendation(
        plot, overrides=body.inputs.model_dump(), crop=body.crop, top_k=body.top_k, log=False
    )


@app.get("/api/plots/{plot_id}/history")
def history(plot_id: str, limit: int = Query(50, ge=1, le=500)):
    _plot_or_404(plot_id)
    return list(reversed(db.list_predictions(plot_id, limit)))


@app.post("/api/predict")
def predict_adhoc(body: AdhocRequest):
    """Stateless prediction (no plot, no sensors): manual values and/or lat/lon for live weather."""
    return pipeline.build_adhoc(body.inputs.model_dump(), body.lat, body.lon, body.crop, body.top_k)
