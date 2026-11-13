"""Live weather from Open-Meteo (free, no API key): https://open-meteo.com

Model features derived from the feed (matching how the training data is defined):
    temperature : mean of the last 7 completed days, degC
    humidity    : mean of the last 7 completed days, %
    rainfall    : total precipitation over the last 30 days, mm

Also returned for the dashboard / irrigation alerts: current conditions and a 7-day forecast.
Results are cached; if the API is unreachable the last cached value is served and flagged stale.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import httpx

from ..config import get_settings

URL = "https://api.open-meteo.com/v1/forecast"

_cache: dict[tuple[float, float], tuple[float, dict]] = {}
_lock = threading.Lock()


class WeatherUnavailable(RuntimeError):
    pass


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def parse_open_meteo(data: dict) -> dict:
    """Pure function (easy to unit-test): raw API JSON -> our payload."""
    daily = data["daily"]
    cur = data.get("current", {})
    today = (cur.get("time") or datetime.now(timezone.utc).isoformat())[:10]
    dates = daily["time"]
    n_past = sum(1 for d in dates if d < today)  # completed days only
    past = slice(max(0, n_past - 30), n_past)
    last7 = slice(max(0, n_past - 7), n_past)
    temp = _mean(daily["temperature_2m_mean"][last7])
    hum = _mean(daily["relative_humidity_2m_mean"][last7])
    rain_vals = [x for x in daily["precipitation_sum"][past] if x is not None]
    rain = sum(rain_vals) if rain_vals else None
    fut = slice(n_past, n_past + 7)
    return {
        "features": {
            "temperature": None if temp is None else round(temp, 2),
            "humidity": None if hum is None else round(hum, 2),
            "rainfall": None if rain is None else round(rain, 1),
        },
        "current": {
            "temperature": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "precipitation": cur.get("precipitation"),
            "wind_speed": cur.get("wind_speed_10m"),
            "time": cur.get("time"),
        },
        "forecast": [
            {
                "date": d,
                "t_max": tx,
                "t_min": tn,
                "rain_mm": r,
                "rain_prob": pp,
            }
            for d, tx, tn, r, pp in zip(
                dates[fut],
                daily.get("temperature_2m_max", [None] * len(dates))[fut],
                daily.get("temperature_2m_min", [None] * len(dates))[fut],
                daily["precipitation_sum"][fut],
                daily.get("precipitation_probability_max", [None] * len(dates))[fut],
            )
        ],
        "history_days": n_past,
    }


def _mock(lat: float, lon: float) -> dict:
    """Plausible fake data for offline development (CA_MOCK_WEATHER=1). Never use in production."""
    import math
    from datetime import date, timedelta

    t0 = date.today()
    days = [(t0 + timedelta(days=i)).isoformat() for i in range(-30, 7)]
    base_t = 30 - abs(lat) * 0.35
    rain = [round(max(0.0, 6 * math.sin(i / 3.0) + 4), 1) for i in range(37)]
    raw = {
        "current": {"time": f"{t0.isoformat()}T12:00", "temperature_2m": round(base_t + 2, 1), "relative_humidity_2m": 68,
                    "precipitation": 0.0, "wind_speed_10m": 9.0},
        "daily": {
            "time": days,
            "temperature_2m_mean": [round(base_t + math.sin(i / 5), 1) for i in range(37)],
            "temperature_2m_max": [round(base_t + 5, 1)] * 37,
            "temperature_2m_min": [round(base_t - 5, 1)] * 37,
            "relative_humidity_2m_mean": [72.0] * 37,
            "precipitation_sum": rain,
            "precipitation_probability_max": [50] * 37,
        },
    }
    out = parse_open_meteo(raw)
    out.update(fetched_at=datetime.now(timezone.utc).isoformat(), source="mock")
    return {**out, "stale": False, "cached": False}


def fetch_weather(lat: float, lon: float, *, force: bool = False) -> dict:
    s = get_settings()
    if s.mock_weather:
        return _mock(lat, lon)
    key = (round(lat, 2), round(lon, 2))  # ~1 km grid; plots close together share a call
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit and not force and now - hit[0] < s.weather_cache_seconds:
        return {**hit[1], "stale": False, "cached": True}
    try:
        r = httpx.get(
            URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                "relative_humidity_2m_mean,precipitation_sum,precipitation_probability_max",
                "past_days": 30,
                "forecast_days": 7,
                "timezone": "auto",
            },
            timeout=s.http_timeout,
        )
        r.raise_for_status()
        payload = parse_open_meteo(r.json())
        payload["fetched_at"] = datetime.now(timezone.utc).isoformat()
        payload["source"] = "open-meteo"
        with _lock:
            _cache[key] = (now, payload)
        return {**payload, "stale": False, "cached": False}
    except Exception as exc:  # network down, rate limit, bad JSON ...
        if hit:  # serve the last good answer rather than failing the whole recommendation
            return {**hit[1], "stale": True, "cached": True, "error": str(exc)}
        raise WeatherUnavailable(str(exc)) from exc


def clear_cache() -> None:
    with _lock:
        _cache.clear()
