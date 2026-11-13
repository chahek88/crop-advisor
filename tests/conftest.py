import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# isolate DB; reuse the trained models in ./models (tests train them if missing)
_tmp = tempfile.mkdtemp(prefix="ca-test-")
os.environ["CA_DB_PATH"] = str(Path(_tmp) / "test.db")
os.environ["CA_API_KEY"] = "test-key"
os.environ["CA_USE_SOILGRIDS"] = "0"

import pytest  # noqa: E402


def fake_open_meteo(today="2026-11-02"):
    """Same shape as the real Open-Meteo response: 30 past days, today, 6 more forecast days."""
    from datetime import date, timedelta

    t0 = date.fromisoformat(today)
    days = [(t0 + timedelta(days=i)).isoformat() for i in range(-30, 7)]
    n = len(days)
    return {
        "current": {"time": f"{today}T10:00", "temperature_2m": 27.0, "relative_humidity_2m": 70, "precipitation": 0.0, "wind_speed_10m": 8.0},
        "daily": {
            "time": days,
            "temperature_2m_mean": [24.0] * n,
            "temperature_2m_max": [30.0] * n,
            "temperature_2m_min": [18.0] * n,
            "relative_humidity_2m_mean": [80.0] * n,
            "precipitation_sum": [7.0] * 30 + [0.0] + [5.0] * 6,  # 30 past days x 7 mm = 210 mm; next 3 days (today incl.) = 10 mm
            "precipitation_probability_max": [40] * n,
        },
    }


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from cropadvisor.main import app
    from cropadvisor.services import weather

    weather.clear_cache()

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return fake_open_meteo()

    monkeypatch.setattr(weather.httpx, "get", lambda *a, **k: R())
    with TestClient(app) as c:
        yield c
