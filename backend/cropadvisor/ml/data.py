"""Dataset loading.

Real data (preferred) -- drop these files into ``data/``:

  data/Crop_recommendation.csv   Kaggle "Crop Recommendation Dataset"
        columns: N,P,K,temperature,humidity,ph,rainfall,label
  data/crop_yield.csv            your own yield table (district / state level)
        columns: crop,N,P,K,temperature,humidity,ph,rainfall,yield_kg_ha[,region]

If a file is missing, a synthetic stand-in with the same schema is generated so the whole
system runs out of the box. The returned ``source`` string says which one was used and is
shown in the UI, so nobody mistakes synthetic numbers for real ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_settings
from .profiles import (
    CROP_PROFILES,
    FEATURE_BOUNDS,
    FEATURES,
    REFERENCE_YIELD_KG_HA,
    REGIONS,
)

CROP_FILE = "Crop_recommendation.csv"
YIELD_FILE = "crop_yield.csv"


def _clip(df: pd.DataFrame) -> pd.DataFrame:
    for f, (lo, hi) in FEATURE_BOUNDS.items():
        df[f] = df[f].clip(lo, hi)
    return df


def synthetic_crop_data(n_per_crop: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for crop, prof in CROP_PROFILES.items():
        cols = {f: rng.normal(m, s, n_per_crop) for f, (m, s) in prof.items()}
        d = pd.DataFrame(cols)
        d["label"] = crop
        parts.append(d)
    return _clip(pd.concat(parts, ignore_index=True))


def synthetic_yield_data(n_per_crop: int = 500, seed: int = 7) -> pd.DataFrame:
    """Yield = reference yield x agronomic suitability x region effect x noise.

    Conditions are sampled with a *wider* spread than the crop-recommendation data so the
    regressor also sees poor sites (that is where yield actually drops).
    """
    rng = np.random.default_rng(seed)
    weights = {"N": 0.14, "P": 0.10, "K": 0.10, "temperature": 0.20, "humidity": 0.10, "ph": 0.14, "rainfall": 0.22}
    region_names = list(REGIONS)
    parts = []
    for crop, prof in CROP_PROFILES.items():
        spread = rng.uniform(0.6, 3.0, n_per_crop)
        d = {}
        for f, (m, s) in prof.items():
            d[f] = rng.normal(m, s * spread)
        d = pd.DataFrame(d)
        region = rng.choice(region_names, n_per_crop)
        d["region"] = region
        d["temperature"] += np.array([REGIONS[r][0] for r in region])
        d["rainfall"] *= np.array([REGIONS[r][1] for r in region])
        d = _clip(d)
        # suitability in (0,1]: Gaussian response around the crop optimum, tolerance = 2.5 x class std
        z2 = sum(weights[f] * ((d[f] - m) / (2.5 * s)) ** 2 for f, (m, s) in prof.items())
        suit = np.exp(-0.5 * z2 / sum(weights.values()) * 3.0)
        mult = np.array([REGIONS[r][2] for r in region])
        noise = rng.lognormal(0, 0.08, n_per_crop)
        d["yield_kg_ha"] = REFERENCE_YIELD_KG_HA[crop] * (0.2 + 0.95 * suit) * mult * noise
        d["crop"] = crop
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def load_crop_data() -> tuple[pd.DataFrame, str]:
    path = get_settings().data_dir / CROP_FILE
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        missing = set(FEATURES + ["label"]) - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        return df[FEATURES + ["label"]].dropna().reset_index(drop=True), f"real:{path.name}"
    return synthetic_crop_data(), "synthetic"


def load_yield_data() -> tuple[pd.DataFrame, str]:
    path = get_settings().data_dir / YIELD_FILE
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        missing = set(FEATURES + ["crop", "yield_kg_ha"]) - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df["crop"] = df["crop"].astype(str).str.strip().str.lower()
        df = df[df["yield_kg_ha"] > 0]
        if "region" not in df.columns:
            df["region"] = "all"
        return df.reset_index(drop=True), f"real:{path.name}"
    return synthetic_yield_data(), "synthetic"
