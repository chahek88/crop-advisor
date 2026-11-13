"""Fake soil-sensor node for development and demos.

Pushes a new reading every few seconds so you can watch the dashboard update live.
Replace this with your real hardware (ESP32 / Arduino / LoRa gateway) -- it only needs to
POST the same JSON to /api/sensors/readings, or publish it to MQTT.

    python simulator/sensor_simulator.py --plot demo-farm --interval 5
    python simulator/sensor_simulator.py --plot demo-farm --profile rice --mqtt-host localhost
    python simulator/sensor_simulator.py --plot demo-farm --drought       # drying-out scenario

Needs only `requests` (and `paho-mqtt` for --mqtt-host).
"""
from __future__ import annotations

import argparse
import json
import random
import time

import requests

PROFILES = {  # typical soil of a plot suited to each crop: N, P, K, pH
    "rice": dict(n=80, p=48, k=40, ph=6.4),
    "maize": dict(n=78, p=48, k=20, ph=6.2),
    "cotton": dict(n=118, p=46, k=20, ph=6.9),
    "chickpea": dict(n=40, p=68, k=80, ph=7.3),
    "banana": dict(n=100, p=82, k=50, ph=6.0),
    "apple": dict(n=21, p=134, k=200, ph=5.9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--key", default="dev-sensor-key")
    ap.add_argument("--plot", required=True)
    ap.add_argument("--device", default="sim-node-1")
    ap.add_argument("--profile", default="rice", choices=PROFILES)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--count", type=int, default=0, help="0 = run forever")
    ap.add_argument("--drought", action="store_true", help="soil moisture steadily falls")
    ap.add_argument("--flaky", action="store_true", help="randomly drop some fields (broken probe)")
    ap.add_argument("--mqtt-host", default="")
    a = ap.parse_args()

    base = PROFILES[a.profile]
    state = dict(base, soil_moisture=45.0, soil_temp=24.0)
    mqtt_client = None
    if a.mqtt_host:
        import paho.mqtt.client as mqtt

        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.connect(a.mqtt_host, 1883)
        mqtt_client.loop_start()

    i = 0
    while a.count == 0 or i < a.count:
        i += 1
        # slow random walk around the profile + sensor noise
        for k in ("n", "p", "k"):
            state[k] = max(0, state[k] + random.gauss((base[k] - state[k]) * 0.1, base[k] * 0.02))
        state["ph"] = min(9, max(4, state["ph"] + random.gauss((base["ph"] - state["ph"]) * 0.1, 0.03)))
        state["soil_temp"] += random.gauss(0, 0.2)
        state["soil_moisture"] = min(95, max(3, state["soil_moisture"] + random.gauss(-1.5 if a.drought else 0, 0.8)))
        msg = {"plot_id": a.plot, "device_id": a.device, **{k: round(v, 2) for k, v in state.items()}}
        if a.flaky:
            for k in list(msg):
                if k not in ("plot_id", "device_id") and random.random() < 0.25:
                    msg.pop(k)
        try:
            if mqtt_client:
                mqtt_client.publish(f"cropadvisor/plots/{a.plot}/readings", json.dumps(msg))
                ok = "mqtt"
            else:
                r = requests.post(f"{a.api}/api/sensors/readings", json=msg, headers={"X-API-Key": a.key}, timeout=10)
                ok = r.status_code
                if r.status_code >= 400:
                    ok = f"{r.status_code} {r.text[:120]}"
            print(f"[{i}] sent {msg} -> {ok}")
        except requests.RequestException as exc:
            print(f"[{i}] send failed: {exc}")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
