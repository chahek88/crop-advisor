"""Central configuration. Everything is overridable with CA_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(f"CA_{name}", default)


@dataclass(frozen=True)
class Settings:
    # storage
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(ROOT / "data" / "cropadvisor.db"))))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", str(ROOT / "models"))))

    # security: required on every write endpoint (sensor ingest, plot create/delete)
    api_key: str = field(default_factory=lambda: _env("API_KEY", "dev-sensor-key"))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))

    # sensors
    # readings older than this are considered "offline" and are NOT fed to the model
    sensor_stale_minutes: float = field(default_factory=lambda: float(_env("SENSOR_STALE_MINUTES", "60")))
    # median of the last few readings inside the window smooths out sensor noise
    sensor_smoothing_readings: int = field(default_factory=lambda: int(_env("SENSOR_SMOOTHING_READINGS", "5")))
    # Multipliers that convert raw sensor units to the units used by the training data
    # (Kaggle crop data is roughly kg/ha; low-cost NPK probes report mg/kg). Calibrate per device!
    n_factor: float = field(default_factory=lambda: float(_env("N_FACTOR", "1.0")))
    p_factor: float = field(default_factory=lambda: float(_env("P_FACTOR", "1.0")))
    k_factor: float = field(default_factory=lambda: float(_env("K_FACTOR", "1.0")))

    # external data
    weather_cache_seconds: int = field(default_factory=lambda: int(_env("WEATHER_CACHE_SECONDS", "600")))
    soil_cache_seconds: int = field(default_factory=lambda: int(_env("SOIL_CACHE_SECONDS", "86400")))
    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "8")))
    # development only: serve synthetic weather instead of calling Open-Meteo (shown as MOCK in the UI)
    mock_weather: bool = field(default_factory=lambda: _env("MOCK_WEATHER", "0") == "1")
    use_soilgrids: bool = field(default_factory=lambda: _env("USE_SOILGRIDS", "1") == "1")

    # demo mode: create a demo plot + history at startup and keep feeding it simulated sensor data
    # (for hosting demos on free tiers whose disk is wiped on every restart)
    demo_mode: bool = field(default_factory=lambda: _env("DEMO_MODE", "0") == "1")
    demo_plot_id: str = field(default_factory=lambda: _env("DEMO_PLOT_ID", "demo-farm"))
    demo_plot_name: str = field(default_factory=lambda: _env("DEMO_PLOT_NAME", "Demo Farm"))
    demo_lat: float = field(default_factory=lambda: float(_env("DEMO_LAT", "30.70")))
    demo_lon: float = field(default_factory=lambda: float(_env("DEMO_LON", "76.72")))
    demo_area_ha: float = field(default_factory=lambda: float(_env("DEMO_AREA_HA", "2.5")))
    demo_interval: float = field(default_factory=lambda: float(_env("DEMO_INTERVAL", "10")))

    # optional MQTT ingest (leave host empty to disable)
    mqtt_host: str = field(default_factory=lambda: _env("MQTT_HOST", ""))
    mqtt_port: int = field(default_factory=lambda: int(_env("MQTT_PORT", "1883")))
    mqtt_username: str = field(default_factory=lambda: _env("MQTT_USERNAME", ""))
    mqtt_password: str = field(default_factory=lambda: _env("MQTT_PASSWORD", ""))
    mqtt_topic: str = field(default_factory=lambda: _env("MQTT_TOPIC", "cropadvisor/plots/+/readings"))


def get_settings() -> Settings:
    return Settings()
