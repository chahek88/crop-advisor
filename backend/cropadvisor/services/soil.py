"""Fallback soil estimate from SoilGrids (ISRIC, 250 m global map) when no pH sensor is online.

Only pH is used: SoilGrids does not provide plant-available N/P/K in the units the crop model
expects, so those stay "missing" (the model handles that) until a real sensor / lab value exists.
"""
from __future__ import annotations

import threading
import time

import httpx

from ..config import get_settings

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
_cache: dict[tuple[float, float], tuple[float, dict | None]] = {}
_lock = threading.Lock()


def parse_soilgrids(data: dict) -> dict | None:
    """Pure function: SoilGrids JSON -> {"ph": float}."""
    for layer in data.get("properties", {}).get("layers", []):
        if layer.get("name") == "phh2o":
            for depth in layer.get("depths", []):
                mean = depth.get("values", {}).get("mean")
                if mean is not None:
                    return {"ph": round(mean / 10.0, 2)}  # stored as pH x 10
    return None


def fetch_soil_ph(lat: float, lon: float) -> dict | None:
    s = get_settings()
    if not s.use_soilgrids:
        return None
    key = (round(lat, 3), round(lon, 3))
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit and now - hit[0] < s.soil_cache_seconds:
        return hit[1]
    try:
        r = httpx.get(
            URL,
            params={"lon": lon, "lat": lat, "property": "phh2o", "depth": "0-5cm", "value": "mean"},
            timeout=s.http_timeout,
        )
        r.raise_for_status()
        res = parse_soilgrids(r.json())
    except Exception:
        return hit[1] if hit else None
    with _lock:
        _cache[key] = (now, res)
    return res
