"""Demo mode (CA_DEMO_MODE=1): a built-in simulated soil sensor.

Free hosting wipes the disk on every restart, so at startup we (re)create a demo plot with 90 minutes of
history and then keep adding a reading every CA_DEMO_INTERVAL seconds. It runs inside the API process
(a thread), so it costs no extra memory. Never enable it with real sensors on the same plot.
"""
from __future__ import annotations

import logging
import random
import threading
from datetime import timedelta

from . import db
from .config import get_settings

log = logging.getLogger("cropadvisor.demo")

_BASE = dict(n=80.0, p=48.0, k=40.0, ph=6.4, soil_moisture=45.0, soil_temp=24.0)  # rice-like soil
_NOISE = dict(n=1.5, p=1.0, k=0.8, ph=0.03, soil_moisture=0.8, soil_temp=0.2)


def _step(state: dict) -> dict:
    """Slow random walk that gently returns to the base values."""
    for k, sd in _NOISE.items():
        state[k] = max(0.0, state[k] + random.gauss((_BASE[k] - state[k]) * 0.05, sd))
    state["ph"] = min(9.0, max(4.0, state["ph"]))
    state["soil_moisture"] = min(95.0, state["soil_moisture"])
    return {k: round(v, 2) for k, v in state.items()}


def seed(history_minutes: int = 90) -> dict:
    """Make sure the demo plot exists and has recent history. Returns the simulator state."""
    s = get_settings()
    state = dict(_BASE)
    if not db.get_plot(s.demo_plot_id):
        db.create_plot(s.demo_plot_id, s.demo_plot_name, s.demo_lat, s.demo_lon, s.demo_area_ha)
    if not db.list_readings(s.demo_plot_id, limit=1):
        now = db.utcnow()
        for i in range(history_minutes):  # one reading per minute, oldest first
            db.insert_reading(s.demo_plot_id, "demo-backfill", now - timedelta(minutes=history_minutes - i), _step(state))
    return state


def _loop(state: dict, stop: threading.Event) -> None:
    s = get_settings()
    n = 0
    while not stop.wait(s.demo_interval):
        try:
            db.insert_reading(s.demo_plot_id, "demo-sensor", db.utcnow(), _step(state))
            n += 1
            if n % 360 == 0:
                db.prune_readings(days=3)  # keep the demo database small
        except Exception:  # e.g. the plot was deleted in the UI: keep running
            log.debug("demo reading skipped", exc_info=True)


def start() -> threading.Event | None:
    if not get_settings().demo_mode:
        return None
    state = seed()
    stop = threading.Event()
    threading.Thread(target=_loop, args=(state, stop), daemon=True, name="demo-sensor").start()
    log.info("demo mode on: plot '%s'", get_settings().demo_plot_id)
    return stop
