"""Weather resilience: 429 retry, NASA POWER fallback, failure cool-down."""
import pytest

from cropadvisor.services import weather


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"Client error '{self.status_code}'")

    def json(self):
        return self._body


def nasa_body():
    from datetime import date, timedelta

    days = [(date.today() - timedelta(days=i)).strftime("%Y%m%d") for i in range(40, 0, -1)]
    # the 3 newest days are not available yet (-999), like the real service
    t = {d: (-999 if i >= 37 else 25.0) for i, d in enumerate(days)}
    h = {d: (-999 if i >= 37 else 60.0) for i, d in enumerate(days)}
    r = {d: (-999 if i >= 37 else 2.0) for i, d in enumerate(days)}
    return {"properties": {"parameter": {"T2M": t, "RH2M": h, "PRECTOTCORR": r}}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    weather.clear_cache()
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    yield
    weather.clear_cache()


def test_parse_nasa_ignores_missing_and_scales_rain():
    from datetime import date

    out = weather.parse_nasa_power(nasa_body(), date.today().isoformat())
    f = out["features"]
    assert f["temperature"] == 25.0 and f["humidity"] == 60.0
    assert f["rainfall"] == 60.0  # 2 mm/day x 30 days
    assert out["forecast"] == [] and out["current"] == {}


def test_falls_back_to_nasa_on_429(monkeypatch):
    calls = []

    def fake_get(url, **k):
        calls.append(url)
        if "open-meteo" in url:
            return Resp(429)
        return Resp(200, nasa_body())

    monkeypatch.setattr(weather.httpx, "get", fake_get)
    out = weather.fetch_weather(31.3, 75.6)
    assert out["source"] == "nasa-power" and out["features"]["temperature"] == 25.0
    assert sum("open-meteo" in c for c in calls) == 2  # first try + one retry


def test_total_failure_then_cooldown(monkeypatch):
    calls = []

    def fake_get(url, **k):
        calls.append(url)
        return Resp(429)

    monkeypatch.setattr(weather.httpx, "get", fake_get)
    with pytest.raises(weather.WeatherUnavailable):
        weather.fetch_weather(31.3, 75.6)
    n = len(calls)
    for _ in range(5):  # dashboard refreshes: no further outgoing calls during the cool-down
        with pytest.raises(weather.WeatherUnavailable):
            weather.fetch_weather(31.3, 75.6)
    assert len(calls) == n


def test_open_meteo_still_preferred_when_healthy(monkeypatch):
    from conftest import fake_open_meteo

    monkeypatch.setattr(weather.httpx, "get", lambda url, **k: Resp(200, fake_open_meteo()))
    assert weather.fetch_weather(31.3, 75.6)["source"] == "open-meteo"
