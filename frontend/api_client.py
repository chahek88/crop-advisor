"""How the Streamlit dashboard talks to the backend.

* Normal mode  : HTTP calls to the FastAPI server (CA_API_URL).
* Embedded mode (CA_MODE=embedded, used by the Docker/Render deployment): the same calls are answered by
  running the backend code inside the dashboard process. One process = one copy of pandas/scikit-learn/the
  models in memory, which is what makes the app fit on a 512 MB host.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import requests

API_URL = os.getenv("CA_API_URL", "http://localhost:8000")
API_KEY = os.getenv("CA_API_KEY", "dev-sensor-key")
EMBEDDED = os.getenv("CA_MODE", "") == "embedded"


class APIError(Exception):
    pass


class Client:
    """HTTP client for the FastAPI backend."""

    def __init__(self, base: str = API_URL, key: str = API_KEY, timeout: float = 25):
        self.base, self.key, self.timeout = base.rstrip("/"), key, timeout

    def _req(self, method: str, path: str, **kw):
        headers = {"X-API-Key": self.key} if method != "GET" else {}
        try:
            r = requests.request(method, f"{self.base}{path}", headers=headers, timeout=self.timeout, **kw)
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach the backend at {self.base} ({exc.__class__.__name__}). Is it running?") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise APIError(f"{r.status_code}: {detail}")
        return None if r.status_code == 204 or not r.content else r.json()

    def health(self):
        return self._req("GET", "/health")

    def model_info(self):
        return self._req("GET", "/api/model/info")

    def plots(self):
        return self._req("GET", "/api/plots")

    def create_plot(self, name, lat, lon, area_ha):
        return self._req("POST", "/api/plots", json={"name": name, "lat": lat, "lon": lon, "area_ha": area_ha})

    def delete_plot(self, plot_id):
        return self._req("DELETE", f"/api/plots/{plot_id}")

    def readings(self, plot_id, limit=300):
        return self._req("GET", f"/api/plots/{plot_id}/readings", params={"limit": limit})

    def recommendation(self, plot_id, top_k=3, crop=None, log=True):
        params = {"top_k": top_k, "log": str(log).lower()}
        if crop:
            params["crop"] = crop
        return self._req("GET", f"/api/plots/{plot_id}/recommendation", params=params)

    def whatif(self, plot_id, inputs: dict, crop=None):
        return self._req("POST", f"/api/plots/{plot_id}/recommendation", json={"inputs": inputs, "crop": crop})

    def history(self, plot_id, limit=100):
        return self._req("GET", f"/api/plots/{plot_id}/history", params={"limit": limit})

    def send_reading(self, plot_id, **values):
        return self._req("POST", "/api/sensors/readings", json={"plot_id": plot_id, "device_id": "dashboard-test", **values})


class LocalClient:
    """Same interface as Client, but runs the backend code in this process (no HTTP, no second server)."""

    def __init__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
        from cropadvisor import db, demo
        from cropadvisor.config import get_settings
        from cropadvisor.ml.predictor import get_predictor
        from cropadvisor.services import pipeline

        self.db, self.pipeline, self.settings = db, pipeline, get_settings()
        db.init_db()
        demo.start()  # no-op unless CA_DEMO_MODE=1
        threading.Thread(target=get_predictor, args=(self.settings.models_dir,), daemon=True).start()

    def _plot(self, plot_id):
        p = self.db.get_plot(plot_id)
        if not p:
            raise APIError(f"404: plot '{plot_id}' not found")
        return p

    def health(self):
        return {"status": "ok", "models_ready": (self.settings.models_dir / "crop_classifier.joblib").exists(), "mqtt": False}

    def model_info(self):
        from cropadvisor.ml.predictor import get_predictor

        return get_predictor(self.settings.models_dir).info()

    def plots(self):
        return self.db.list_plots()

    def create_plot(self, name, lat, lon, area_ha):
        from cropadvisor.schemas import PlotIn

        try:
            p = PlotIn(name=name, lat=lat, lon=lon, area_ha=area_ha)
            return self.db.create_plot(None, p.name, p.lat, p.lon, p.area_ha)
        except ValueError as exc:
            raise APIError(f"409: {exc}") from exc

    def delete_plot(self, plot_id):
        if not self.db.delete_plot(plot_id):
            raise APIError("404: not found")

    def readings(self, plot_id, limit=300):
        self._plot(plot_id)
        return list(reversed(self.db.list_readings(plot_id, limit=limit)))

    def _recommend(self, plot_id, **kw):
        plot = self._plot(plot_id)
        try:
            return self.pipeline.build_recommendation(plot, **kw)
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"500: {exc}") from exc

    def recommendation(self, plot_id, top_k=3, crop=None, log=True):
        return self._recommend(plot_id, crop=crop, top_k=top_k, log=log)

    def whatif(self, plot_id, inputs: dict, crop=None):
        return self._recommend(plot_id, overrides=inputs, crop=crop, top_k=3, log=False)

    def history(self, plot_id, limit=100):
        self._plot(plot_id)
        return list(reversed(self.db.list_predictions(plot_id, limit)))

    def send_reading(self, plot_id, **values):
        from cropadvisor.schemas import SensorReading

        self._plot(plot_id)
        try:
            r = SensorReading(plot_id=plot_id, device_id="dashboard-test", **values)
        except ValueError as exc:
            raise APIError(f"422: {exc}") from exc
        return {"id": self.db.insert_reading(plot_id, r.device_id, self.db.utcnow(), r.model_dump()), "plot_id": plot_id}


def make_client():
    return LocalClient() if EMBEDDED else Client()
