"""Glue: sensors + live weather + soil fallback -> features -> model -> alerts.

Priority for every model input (highest first):
    manual override  >  fresh sensor reading  >  live weather (weather features) >  SoilGrids (pH)  >  missing
"""
from __future__ import annotations

import statistics
from datetime import timedelta

from .. import db
from ..config import get_settings
from ..ml.predictor import get_predictor
from . import soil, weather

SENSOR_TO_FEATURE = {"n": "N", "p": "P", "k": "K", "ph": "ph"}


def sensor_snapshot(plot_id: str) -> dict:
    """Fresh, smoothed sensor values + online/offline status."""
    s = get_settings()
    now = db.utcnow()
    latest = db.latest_reading(plot_id)
    status = {"online": False, "last_seen": None, "age_minutes": None, "device_id": None, "values": {}, "n_readings": 0}
    if latest:
        age = (now - _parse(latest["ts"])).total_seconds() / 60
        status.update(last_seen=latest["ts"], age_minutes=round(age, 1), device_id=latest["device_id"])
    rows = db.list_readings(plot_id, limit=s.sensor_smoothing_readings, since=now - timedelta(minutes=s.sensor_stale_minutes))
    status["n_readings"] = len(rows)
    if rows:
        status["online"] = True
        for f in db.READING_FIELDS:
            vals = [r[f] for r in rows if r[f] is not None]
            if vals:
                status["values"][f] = round(statistics.median(vals), 3)  # median = robust to spikes
    return status


def _parse(ts: str):
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _calibrate(values: dict) -> dict:
    s = get_settings()
    out = dict(values)
    for k, factor in (("n", s.n_factor), ("p", s.p_factor), ("k", s.k_factor)):
        if k in out:
            out[k] = out[k] * factor
    return out


def _alerts(sensor: dict, wx: dict | None, wx_error: str | None) -> list[dict]:
    a = []
    if not sensor["last_seen"]:
        a.append({"level": "info", "message": "No sensor data received yet for this plot. Using weather/soil-map data only."})
    elif not sensor["online"]:
        a.append({"level": "warning", "message": f"Sensors offline: last reading {sensor['age_minutes']:.0f} min ago. Soil values are not used."})
    if wx_error:
        a.append({"level": "warning", "message": f"Live weather unavailable ({wx_error[:80]}). Weather inputs are missing."})
    elif wx and wx.get("stale"):
        a.append({"level": "info", "message": "Weather service unreachable; showing the last cached data."})

    v = sensor["values"]
    rain_3d = None
    if wx and wx.get("forecast"):
        rain_3d = sum((d["rain_mm"] or 0) for d in wx["forecast"][:3])
    m = v.get("soil_moisture")
    if m is not None:
        if m < 20:
            if rain_3d is not None and rain_3d >= 8:
                a.append({"level": "info", "message": f"Soil is dry ({m:.0f}%) but ~{rain_3d:.0f} mm rain is forecast within 3 days: hold off irrigating."})
            else:
                a.append({"level": "critical", "message": f"Soil moisture is low ({m:.0f}%) and little rain is forecast: irrigate."})
        elif m > 80:
            a.append({"level": "warning", "message": f"Soil moisture is very high ({m:.0f}%): risk of waterlogging, check drainage."})
    st = v.get("soil_temp")
    if st is not None and (st < 10 or st > 38):
        a.append({"level": "warning", "message": f"Soil temperature is {st:.0f} °C, outside the comfortable range for most crops."})
    return a


LOG_EVERY_SECONDS = 300


def _should_log(plot_id: str, top_crop: str) -> bool:
    """The dashboard polls every few seconds; only store a history row when it is informative."""
    last = db.list_predictions(plot_id, limit=1)
    if not last:
        return True
    changed = last[0]["top_crop"] != top_crop
    return changed or (db.utcnow() - _parse(last[0]["ts"])).total_seconds() >= LOG_EVERY_SECONDS


def build_recommendation(
    plot: dict,
    *,
    overrides: dict | None = None,
    crop: str | None = None,
    top_k: int = 3,
    log: bool = True,
) -> dict:
    s = get_settings()
    predictor = get_predictor(s.models_dir)

    sensor = sensor_snapshot(plot["id"])
    sensor_vals = _calibrate(sensor["values"])

    wx, wx_error = None, None
    try:
        wx = weather.fetch_weather(plot["lat"], plot["lon"])
    except weather.WeatherUnavailable as exc:
        wx_error = str(exc) or "unreachable"

    features: dict[str, float | None] = {}
    sources: dict[str, str | None] = {}
    for sk, mk in SENSOR_TO_FEATURE.items():
        if sk in sensor_vals:
            features[mk], sources[mk] = sensor_vals[sk], "sensor"
    if wx:
        for k, v in wx["features"].items():
            if v is not None:
                features[k], sources[k] = v, "weather"
    if "ph" not in features:
        sg = soil.fetch_soil_ph(plot["lat"], plot["lon"])
        if sg:
            features["ph"], sources["ph"] = sg["ph"], "soil-map"
    for k, v in (overrides or {}).items():
        if v is not None:
            features[k], sources[k] = v, "manual"

    result = predictor.recommend(features, top_k=top_k, crop=crop)
    result.update(
        plot=plot,
        sources={k: sources.get(k) for k in predictor.profile["features"]},
        sensor={k: v for k, v in sensor.items() if k != "values"} | {"values": sensor["values"]},
        weather=None if not wx else {k: wx[k] for k in ("current", "forecast", "stale", "fetched_at", "source") if k in wx},
        alerts=_alerts(sensor, wx, wx_error),
        generated_at=db.iso(db.utcnow()),
    )
    if log and _should_log(plot["id"], result["recommendations"][0]["crop"]):
        db.log_prediction(plot["id"], {k: result[k] for k in ("recommendations", "confidence", "inputs", "sources")})
    return result


def build_adhoc(inputs: dict, lat: float | None, lon: float | None, crop: str | None, top_k: int) -> dict:
    """No registered plot: manual values, optionally topped up with live weather for lat/lon."""
    s = get_settings()
    predictor = get_predictor(s.models_dir)
    features = {k: v for k, v in inputs.items() if v is not None}
    sources = {k: "manual" for k in features}
    wx, wx_error = None, None
    if lat is not None and lon is not None:
        try:
            wx = weather.fetch_weather(lat, lon)
            for k, v in wx["features"].items():
                if k not in features and v is not None:
                    features[k], sources[k] = v, "weather"
        except weather.WeatherUnavailable as exc:
            wx_error = str(exc) or "unreachable"
        if "ph" not in features:
            sg = soil.fetch_soil_ph(lat, lon)
            if sg:
                features["ph"], sources["ph"] = sg["ph"], "soil-map"
    result = predictor.recommend(features, top_k=top_k, crop=crop)
    result.update(sources={k: sources.get(k) for k in predictor.profile["features"]}, weather_error=wx_error,
                  generated_at=db.iso(db.utcnow()))
    return result
