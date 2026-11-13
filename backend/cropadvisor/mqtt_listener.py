"""Optional MQTT ingest. Enabled when CA_MQTT_HOST is set.

Sensors / gateways publish JSON to   cropadvisor/plots/<plot_id>/readings
    {"device_id": "node-7", "n": 41, "p": 22, "k": 180, "ph": 6.4, "soil_moisture": 31.5, "soil_temp": 24.1}
(``plot_id`` may be in the topic or in the payload; ``timestamp`` is optional.)
"""
from __future__ import annotations

import json
import logging

from . import db
from .config import get_settings
from .schemas import SensorReading

log = logging.getLogger("cropadvisor.mqtt")


def handle_message(topic: str, payload: bytes) -> bool:
    """Parse + store one message. Returns True when stored. Never raises on bad input."""
    try:
        data = json.loads(payload)
        parts = topic.split("/")
        if "plot_id" not in data and len(parts) >= 3:
            data["plot_id"] = parts[-2]
        reading = SensorReading(**data)
        if not db.get_plot(reading.plot_id):
            log.warning("mqtt: unknown plot %s", reading.plot_id)
            return False
        db.insert_reading(reading.plot_id, reading.device_id, reading.timestamp or db.utcnow(), reading.model_dump())
        return True
    except Exception as exc:
        log.warning("mqtt: bad message on %s: %s", topic, exc)
        return False


def start():
    s = get_settings()
    if not s.mqtt_host:
        return None
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if s.mqtt_username:
        client.username_pw_set(s.mqtt_username, s.mqtt_password)

    def on_connect(c, userdata, flags, reason_code, properties=None):
        log.info("mqtt connected (%s); subscribing to %s", reason_code, s.mqtt_topic)
        c.subscribe(s.mqtt_topic)

    client.on_connect = on_connect
    client.on_message = lambda c, u, m: handle_message(m.topic, m.payload)
    client.reconnect_delay_set(1, 30)
    client.connect_async(s.mqtt_host, s.mqtt_port, keepalive=60)
    client.loop_start()
    return client
