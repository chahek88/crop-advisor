import math
from datetime import datetime, timedelta, timezone

import pytest

from conftest import fake_open_meteo

H = {"X-API-Key": "test-key"}


# ------------------------------------------------------------------ pure parsers
def test_weather_parser_features():
    from cropadvisor.services.weather import parse_open_meteo

    out = parse_open_meteo(fake_open_meteo())
    f = out["features"]
    assert f["temperature"] == 24.0
    assert f["humidity"] == 80.0
    assert f["rainfall"] == 210.0  # 30 completed days x 7 mm; today and forecast excluded
    assert len(out["forecast"]) == 7 and out["forecast"][0]["rain_mm"] == 0.0


def test_weather_parser_handles_nulls():
    from cropadvisor.services.weather import parse_open_meteo

    d = fake_open_meteo()
    d["daily"]["precipitation_sum"][:5] = [None] * 5
    assert parse_open_meteo(d)["features"]["rainfall"] == 25 * 7.0


def test_soilgrids_parser():
    from cropadvisor.services.soil import parse_soilgrids

    data = {"properties": {"layers": [{"name": "phh2o", "depths": [{"values": {"mean": 64}}]}]}}
    assert parse_soilgrids(data) == {"ph": 6.4}
    assert parse_soilgrids({}) is None


# ------------------------------------------------------------------ model
def test_predictor_handles_missing_and_empty():
    from cropadvisor.config import get_settings
    from cropadvisor.ml.predictor import get_predictor

    p = get_predictor(get_settings().models_dir)
    full = dict(N=80, P=48, K=40, temperature=24, humidity=82, ph=6.4, rainfall=230)
    r = p.recommend(full)
    assert r["recommendations"][0]["crop"] == "rice" and r["confidence"] == "high"
    y = r["recommendations"][0]["yield"]
    assert y["p10"] < y["p50"] < y["p90"]

    part = p.recommend({**full, "N": None, "K": None})
    assert part["recommendations"][0]["crop"] == "rice"
    assert set(part["missing"]) == {"N", "K"} and part["confidence"] != "high"

    empty = p.recommend({})  # must not crash even with zero inputs
    assert empty["confidence"] == "low" and len(empty["missing"]) == 7
    assert math.isclose(sum(c["probability"] for c in p.recommend({}, top_k=10)["recommendations"]), 1.0, abs_tol=0.5)


# ------------------------------------------------------------------ API flow
def test_auth_required_for_writes(client):
    assert client.post("/api/plots", json={"name": "x", "lat": 1, "lon": 1}).status_code == 401
    assert client.post("/api/sensors/readings", json={"plot_id": "x", "ph": 6}).status_code == 401


def test_end_to_end_live_flow(client):
    r = client.post("/api/plots", json={"name": "Test Farm", "lat": 28.6, "lon": 77.2}, headers=H)
    assert r.status_code == 201
    pid = r.json()["id"]
    assert pid == "test-farm"
    assert client.post("/api/plots", json={"name": "Test Farm", "lat": 1, "lon": 1}, headers=H).status_code == 409

    # no sensors yet -> weather-only recommendation with an informative alert
    rec = client.get(f"/api/plots/{pid}/recommendation").json()
    assert rec["sensor"]["online"] is False
    assert set(rec["missing"]) >= {"N", "P", "K"}
    assert rec["sources"]["temperature"] == "weather"
    assert any("No sensor data" in a["message"] for a in rec["alerts"])

    # sensors come online
    for _ in range(3):
        r = client.post("/api/sensors/readings", headers=H, json={
            "plot_id": pid, "device_id": "n1", "n": 80, "p": 48, "k": 40, "ph": 6.4, "soil_moisture": 15, "soil_temp": 24})
        assert r.status_code == 201
    rec = client.get(f"/api/plots/{pid}/recommendation").json()
    assert rec["sensor"]["online"] is True
    assert rec["sources"]["N"] == "sensor" and rec["missing"] == []
    assert rec["recommendations"][0]["crop"] == "rice"
    # dry soil + 10 mm rain forecast within 3 days -> hold off, not critical
    assert any("hold off" in a["message"] for a in rec["alerts"])

    # what-if override wins over sensors
    wi = client.post(f"/api/plots/{pid}/recommendation", json={"inputs": {"N": 5}}).json()
    assert wi["sources"]["N"] == "manual" and wi["inputs"]["N"] == 5

    assert len(client.get(f"/api/plots/{pid}/readings").json()) == 3
    hist = client.get(f"/api/plots/{pid}/history").json()
    assert 1 <= len(hist) <= 2  # throttled: only crop changes / 5-minute ticks are stored; what-if never is


def test_stale_sensor_is_not_used(client):
    pid = client.post("/api/plots", json={"name": "Old Sensor", "lat": 10, "lon": 10}, headers=H).json()["id"]
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    client.post("/api/sensors/readings", headers=H, json={"plot_id": pid, "n": 99, "p": 99, "k": 99, "ph": 7, "timestamp": old})
    rec = client.get(f"/api/plots/{pid}/recommendation").json()
    assert rec["sensor"]["online"] is False and rec["sources"]["N"] is None
    assert any("offline" in a["message"] for a in rec["alerts"])


def test_validation_and_unknown_plot(client):
    assert client.post("/api/sensors/readings", headers=H, json={"plot_id": "nope", "ph": 6}).status_code == 404
    assert client.post("/api/sensors/readings", headers=H, json={"plot_id": "nope"}).status_code == 422  # no values
    assert client.post("/api/sensors/readings", headers=H, json={"plot_id": "nope", "ph": 40}).status_code == 422


def test_adhoc_predict(client):
    r = client.post("/api/predict", json={"inputs": {"N": 118, "P": 46, "K": 20, "ph": 6.9}, "lat": 21.1, "lon": 79.0}).json()
    assert r["sources"]["rainfall"] == "weather" and r["sources"]["N"] == "manual"
    assert r["recommendations"]


def test_weather_outage_degrades_gracefully(client, monkeypatch):
    from cropadvisor.services import weather

    weather.clear_cache()
    monkeypatch.setattr(weather.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network down")))
    pid = client.post("/api/plots", json={"name": "Offline Wx", "lat": 5, "lon": 5}, headers=H).json()["id"]
    client.post("/api/sensors/readings", headers=H, json={"plot_id": pid, "n": 80, "p": 48, "k": 40, "ph": 6.4})
    rec = client.get(f"/api/plots/{pid}/recommendation")
    assert rec.status_code == 200  # still answers, from soil data alone
    body = rec.json()
    assert "rainfall" in body["missing"] and any("weather unavailable" in a["message"].lower() for a in body["alerts"])


def test_mqtt_message_handler(client):
    from cropadvisor import mqtt_listener

    pid = client.post("/api/plots", json={"name": "Mqtt Plot", "lat": 1, "lon": 1}, headers=H).json()["id"]
    assert mqtt_listener.handle_message(f"cropadvisor/plots/{pid}/readings", b'{"ph": 6.1, "n": 30}')
    assert not mqtt_listener.handle_message("cropadvisor/plots/ghost/readings", b'{"ph": 6.1}')
    assert not mqtt_listener.handle_message(f"cropadvisor/plots/{pid}/readings", b"not json")
    assert client.get(f"/api/plots/{pid}/sensor").json()["values"]["ph"] == 6.1


def test_irrigation_alert_when_dry_and_no_rain():
    from cropadvisor.services.pipeline import _alerts

    sensor = {"last_seen": "x", "online": True, "age_minutes": 1, "values": {"soil_moisture": 12}}
    dry = {"forecast": [{"rain_mm": 0}] * 7}
    wet = {"forecast": [{"rain_mm": 6}] * 7}
    assert _alerts(sensor, dry, None)[0]["level"] == "critical"
    assert "hold off" in _alerts(sensor, wet, None)[0]["message"]
    sensor["values"]["soil_moisture"] = 90
    assert "waterlogging" in _alerts(sensor, dry, None)[0]["message"]


def test_demo_mode_seeds_plot_and_history(client):
    from cropadvisor import db, demo

    demo.seed(history_minutes=30)
    assert db.get_plot("demo-farm") is not None
    rows = db.list_readings("demo-farm", limit=100)
    assert len(rows) == 30
    demo.seed(history_minutes=30)  # idempotent: second call adds nothing
    assert len(db.list_readings("demo-farm", limit=100)) == 30
    rec = client.get("/api/plots/demo-farm/recommendation").json()
    assert rec["sensor"]["online"] is True and rec["sources"]["N"] == "sensor"


def test_embedded_client_matches_the_api(client):
    """The dashboard's in-process client (used in the hosted deployment) behaves like the HTTP client."""
    from api_client import APIError, LocalClient

    c = LocalClient()
    p = c.create_plot("Embedded Farm", 30.7, 76.7, 2.0)
    assert p["id"] == "embedded-farm"
    with pytest.raises(APIError, match="409"):
        c.create_plot("Embedded Farm", 30.7, 76.7, 2.0)
    with pytest.raises(APIError, match="404"):
        c.recommendation("ghost")
    with pytest.raises(APIError, match="422"):
        c.send_reading("embedded-farm", ph=99)
    for _ in range(3):
        c.send_reading("embedded-farm", n=80, p=48, k=40, ph=6.4, soil_moisture=30)
    rec = c.recommendation("embedded-farm")
    assert rec["sensor"]["online"] and rec["sources"]["N"] == "sensor" and rec["recommendations"]
    assert c.whatif("embedded-farm", {"N": 5})["inputs"]["N"] == 5
    assert len(c.readings("embedded-farm")) == 3 and c.history("embedded-farm")
    assert c.health()["models_ready"] and "classes" in c.model_info()
    c.delete_plot("embedded-farm")
    assert all(x["id"] != "embedded-farm" for x in c.plots())
