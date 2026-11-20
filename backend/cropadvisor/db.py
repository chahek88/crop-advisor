"""Tiny SQLite layer (stdlib only). Swap for Postgres/TimescaleDB when you outgrow it."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS plots (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    area_ha REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plot_id TEXT NOT NULL REFERENCES plots(id) ON DELETE CASCADE,
    device_id TEXT,
    ts TEXT NOT NULL,
    received_at TEXT NOT NULL,
    n REAL, p REAL, k REAL, ph REAL,
    soil_moisture REAL, soil_temp REAL, ec REAL
);
CREATE INDEX IF NOT EXISTS idx_readings_plot_ts ON readings(plot_id, ts DESC);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plot_id TEXT NOT NULL REFERENCES plots(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    top_crop TEXT,
    probability REAL,
    yield_p50 REAL,
    confidence TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_plot_ts ON predictions(plot_id, ts DESC);
"""

READING_FIELDS = ("n", "p", "k", "ph", "soil_moisture", "soil_temp", "ec")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "plot"


# ---------------------------------------------------------------- plots
def create_plot(plot_id: str | None, name: str, lat: float, lon: float, area_ha: float) -> dict:
    pid = plot_id or slugify(name)
    with connect() as con:
        if con.execute("SELECT 1 FROM plots WHERE id=?", (pid,)).fetchone():
            raise ValueError(f"plot '{pid}' already exists")
        con.execute(
            "INSERT INTO plots(id,name,lat,lon,area_ha,created_at) VALUES(?,?,?,?,?,?)",
            (pid, name, lat, lon, area_ha, iso(utcnow())),
        )
    return get_plot(pid)


def get_plot(plot_id: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM plots WHERE id=?", (plot_id,)).fetchone()
    return dict(row) if row else None


def list_plots() -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM plots ORDER BY created_at")]


def delete_plot(plot_id: str) -> bool:
    with connect() as con:
        return con.execute("DELETE FROM plots WHERE id=?", (plot_id,)).rowcount > 0


# ---------------------------------------------------------------- readings
def insert_reading(plot_id: str, device_id: str | None, ts: datetime, values: dict) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO readings(plot_id,device_id,ts,received_at,n,p,k,ph,soil_moisture,soil_temp,ec)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (plot_id, device_id, iso(ts), iso(utcnow()), *[values.get(f) for f in READING_FIELDS]),
        )
        return cur.lastrowid


def list_readings(plot_id: str, limit: int = 200, since: datetime | None = None) -> list[dict]:
    q, args = "SELECT * FROM readings WHERE plot_id=?", [plot_id]
    if since:
        q += " AND ts>=?"
        args.append(iso(since))
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with connect() as con:
        return [dict(r) for r in con.execute(q, args)]


def latest_reading(plot_id: str) -> dict | None:
    rows = list_readings(plot_id, limit=1)
    return rows[0] if rows else None


def prune_readings(days: int = 90) -> int:
    with connect() as con:
        return con.execute("DELETE FROM readings WHERE ts<?", (iso(utcnow() - timedelta(days=days)),)).rowcount


# ---------------------------------------------------------------- predictions
def log_prediction(plot_id: str, result: dict) -> None:
    top = result["recommendations"][0]
    y = (top.get("yield") or {}).get("p50")
    with connect() as con:
        con.execute(
            "INSERT INTO predictions(plot_id,ts,top_crop,probability,yield_p50,confidence,payload) VALUES(?,?,?,?,?,?,?)",
            (plot_id, iso(utcnow()), top["crop"], top["probability"], y, result["confidence"], json.dumps(result)),
        )


def list_predictions(plot_id: str, limit: int = 50) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT id,ts,top_crop,probability,yield_p50,confidence FROM predictions WHERE plot_id=? ORDER BY ts DESC LIMIT ?",
            (plot_id, limit),
        )
        return [dict(r) for r in rows]
