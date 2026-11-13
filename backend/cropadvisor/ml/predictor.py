"""Inference: crop ranking + yield estimate + human-readable explanation.

Every feature may be ``None`` / NaN (sensor offline, API down). The models were trained with
missing-value augmentation, and the response says how much to trust the result.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .profiles import ADVICE, FEATURE_LABELS, FEATURES


def _clean(features: dict) -> dict[str, float]:
    out = {}
    for f in FEATURES:
        v = features.get(f)
        try:
            v = float(v) if v is not None else math.nan
        except (TypeError, ValueError):
            v = math.nan
        out[f] = v
    return out


class Predictor:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.clf = joblib.load(self.models_dir / "crop_classifier.joblib")
        y = joblib.load(self.models_dir / "yield_models.joblib")
        self.yield_models: dict = y["models"]
        self.yield_crops: list[str] = y["crops"]
        self.interval_scale: float = float(y.get("interval_scale", 1.0))
        self.profile: dict = json.loads((self.models_dir / "profile.json").read_text())
        self.clf.n_jobs = 1  # single-row predictions: no thread pool needed
        self.classes = list(self.clf.classes_)

    # ------------------------------------------------------------------ yield
    def estimate_yield(self, crop: str, features: dict) -> dict | None:
        """Yield for a given crop in kg/ha: median and an (approximate) 80% interval."""
        if crop not in self.yield_crops:
            return None
        f = _clean(features)
        row = pd.DataFrame([[float(self.yield_crops.index(crop))] + [f[k] for k in FEATURES]], columns=["crop_idx"] + FEATURES)
        p = {k: float(m.predict(row)[0]) for k, m in self.yield_models.items()}
        mid = p["p50"]
        lo = mid - self.interval_scale * max(mid - p["p10"], 0.0)
        hi = mid + self.interval_scale * max(p["p90"] - mid, 0.0)
        return {
            "p10": round(float(np.exp(lo)), 0),
            "p50": round(float(np.exp(mid)), 0),
            "p90": round(float(np.exp(hi)), 0),
            "unit": "kg/ha",
        }

    # ------------------------------------------------------------------ crop ranking
    def _proba(self, f: dict) -> np.ndarray:
        return self.clf.predict_proba(pd.DataFrame([[f[k] for k in FEATURES]], columns=FEATURES))[0]

    def rank_crops(self, features: dict, top_k: int = 3) -> list[tuple[str, float]]:
        proba = self._proba(_clean(features))
        order = np.argsort(proba)[::-1][:top_k]
        return [(self.classes[i], float(proba[i])) for i in order]

    # ------------------------------------------------------------------ explanation
    def explain(self, crop: str, features: dict) -> dict:
        f = _clean(features)
        rng = self.profile["ranges"].get(crop, {})
        base = self._proba(f)[self.classes.index(crop)]

        status, advice = [], []
        for k in FEATURES:
            v = f[k]
            r = rng.get(k)
            if math.isnan(v):
                status.append({"feature": k, "label": FEATURE_LABELS[k], "value": None, "status": "missing",
                               "ideal_low": r and r["p10"], "ideal_high": r and r["p90"], "influence": 0.0})
                continue
            # influence = how much the crop's probability falls if we pretend this input is unknown
            g = dict(f)
            g[k] = math.nan
            influence = float(base - self._proba(g)[self.classes.index(crop)])
            st = "ok"
            if r:
                span = max(r["p90"] - r["p10"], 1e-6)
                margin = 0.15 * span  # small tolerance outside the p10-p90 band
                if v < r["p10"] - margin:
                    st = "low"
                elif v > r["p90"] + margin:
                    st = "high"
                if st != "ok":
                    advice.append(ADVICE[k][st].format(crop=crop))
            status.append({"feature": k, "label": FEATURE_LABELS[k], "value": round(v, 2), "status": st,
                           "ideal_low": r and round(r["p10"], 2), "ideal_high": r and round(r["p90"], 2),
                           "influence": round(influence, 4)})

        ok = [s for s in status if s["status"] == "ok"]
        off = [s for s in status if s["status"] in ("low", "high")]
        drivers = sorted((s for s in ok), key=lambda s: -s["influence"])[:2]
        parts = []
        if drivers:
            parts.append("Best matches for " + crop + ": " + ", ".join(d["label"] for d in drivers) + ".")
        if off:
            parts.append("Outside the usual range: " + ", ".join(f"{o['label']} ({o['status']})" for o in off) + ".")
        if not parts:
            parts.append("Not enough data to say much — add more sensor readings.")
        return {"crop": crop, "summary": " ".join(parts), "features": status, "advice": advice}

    # ------------------------------------------------------------------ main entry
    def recommend(self, features: dict, top_k: int = 3, crop: str | None = None) -> dict:
        f = _clean(features)
        missing = [k for k in FEATURES if math.isnan(f[k])]
        ranked = self.rank_crops(f, top_k=top_k)
        recs = [{"crop": c, "probability": round(p, 4), "yield": self.estimate_yield(c, f)} for c, p in ranked]
        target = crop or recs[0]["crop"]

        n_avail = len(FEATURES) - len(missing)
        top_p = recs[0]["probability"]
        if n_avail <= 3 or top_p < 0.45:
            confidence = "low"
        elif len(missing) >= 2 or top_p < 0.75:
            confidence = "medium"
        else:
            confidence = "high"
        notes = []
        if missing:
            notes.append("Missing inputs: " + ", ".join(FEATURE_LABELS[m] for m in missing) + ". The estimate is less reliable.")
        if n_avail > 3 and top_p < 0.45:
            notes.append("No crop is a strong match: conditions sit between the profiles of several crops, so treat this as a weak suggestion.")
        if all(math.isnan(f[k]) for k in ("N", "P", "K", "ph")):
            notes.append("No soil data at all — the recommendation is based on weather only.")
        if all(math.isnan(f[k]) for k in ("temperature", "humidity", "rainfall")):
            notes.append("No weather data — the recommendation is based on soil only.")

        return {
            "recommendations": recs,
            "explanation": self.explain(target, f),
            "inputs": {k: (None if math.isnan(v) else round(v, 3)) for k, v in f.items()},
            "missing": missing,
            "confidence": confidence,
            "notes": notes,
            "model": {
                "data_source": self.profile["data_source"],
                "trained_at": self.profile["trained_at"],
            },
        }

    def info(self) -> dict:
        return {
            "classes": self.classes,
            "features": FEATURES,
            "data_source": self.profile["data_source"],
            "trained_at": self.profile["trained_at"],
            "feature_importance": self.profile["feature_importance"],
            "metrics": self.profile["metrics"],
        }


_lock = threading.Lock()
_instance: Predictor | None = None


def get_predictor(models_dir: Path) -> Predictor:
    """Lazy singleton; trains automatically on first run if no models exist."""
    global _instance
    with _lock:
        if _instance is None:
            if not (Path(models_dir) / "crop_classifier.joblib").exists():
                from . import train

                train.main()
            _instance = Predictor(models_dir)
        return _instance


def reload_predictor(models_dir: Path) -> Predictor:
    global _instance
    with _lock:
        _instance = Predictor(models_dir)
        return _instance
