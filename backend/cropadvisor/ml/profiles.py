"""Agronomic reference profiles.

Used ONLY to (a) generate a synthetic stand-in dataset when the real Kaggle files are not
present, and (b) provide reference yields. The per-crop means/stds follow the well-known
Kaggle "Crop Recommendation" dataset (N, P, K, temperature degC, humidity %, pH, rainfall mm).

If you drop the real CSVs into ``data/`` (see README) these numbers are never used for training.
"""
from __future__ import annotations

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SOIL_FEATURES = ["N", "P", "K", "ph"]
WEATHER_FEATURES = ["temperature", "humidity", "rainfall"]

FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall, 30 d (mm)",
}

FEATURE_BOUNDS = {
    "N": (0, 160),
    "P": (0, 160),
    "K": (0, 250),
    "temperature": (5, 45),
    "humidity": (10, 100),
    "ph": (3.5, 9.5),
    "rainfall": (0, 320),
}

# crop: (N, P, K, temperature, humidity, ph, rainfall) as (mean, std)
_P = lambda *pairs: dict(zip(FEATURES, pairs))  # noqa: E731

CROP_PROFILES = {
    "rice": _P((80, 12), (48, 8), (40, 5), (23.7, 2), (82, 3), (6.4, 0.8), (236, 30)),
    "maize": _P((78, 15), (48, 10), (20, 3), (22.4, 2.5), (65, 5), (6.2, 0.6), (85, 15)),
    "chickpea": _P((40, 8), (68, 8), (80, 5), (18.9, 1.5), (17, 3), (7.3, 0.8), (80, 10)),
    "kidneybeans": _P((21, 8), (68, 8), (20, 3), (20, 2), (21, 3), (5.7, 0.4), (106, 15)),
    "pigeonpeas": _P((21, 8), (68, 8), (20, 3), (27.7, 3), (48, 5), (5.8, 0.6), (149, 25)),
    "mothbeans": _P((21, 8), (48, 8), (20, 3), (28.2, 3), (53, 5), (6.8, 1.0), (51, 10)),
    "mungbean": _P((21, 8), (47, 8), (20, 3), (28.5, 2), (85, 3), (6.7, 0.4), (48, 8)),
    "blackgram": _P((40, 8), (67, 8), (19, 3), (30, 2), (65, 3), (7.1, 0.5), (68, 8)),
    "lentil": _P((19, 8), (68, 8), (19, 3), (24.5, 3), (64, 3), (6.9, 0.5), (46, 8)),
    "pomegranate": _P((19, 8), (19, 8), (40, 3), (21.8, 2), (90, 3), (6.4, 0.6), (108, 15)),
    "banana": _P((100, 10), (82, 10), (50, 3), (27.4, 1.5), (80, 3), (6.0, 0.5), (105, 15)),
    "mango": _P((20, 8), (28, 8), (30, 3), (31.2, 2), (50, 3), (5.8, 0.6), (95, 15)),
    "grapes": _P((23, 8), (132, 8), (200, 5), (23.8, 5), (82, 3), (6.0, 0.5), (70, 10)),
    "watermelon": _P((100, 10), (17, 5), (50, 3), (25.6, 2), (85, 3), (6.5, 0.4), (50, 8)),
    "muskmelon": _P((100, 10), (17, 5), (50, 3), (28.7, 2), (92, 3), (6.4, 0.4), (25, 5)),
    "apple": _P((21, 8), (134, 8), (200, 5), (22.6, 2), (92, 3), (5.9, 0.4), (113, 10)),
    "orange": _P((20, 8), (16, 5), (10, 3), (22.8, 3), (92, 3), (7.0, 0.7), (110, 12)),
    "papaya": _P((50, 8), (59, 8), (50, 5), (33.7, 3), (92, 3), (6.7, 0.4), (142, 20)),
    "coconut": _P((22, 8), (17, 5), (31, 3), (27.4, 1.5), (95, 2), (6.0, 0.5), (175, 25)),
    "cotton": _P((118, 10), (46, 8), (20, 3), (24, 2), (80, 3), (6.9, 0.5), (80, 12)),
    "jute": _P((78, 12), (47, 8), (40, 3), (25, 2), (80, 3), (6.7, 0.5), (175, 25)),
    "coffee": _P((101, 10), (28, 8), (30, 3), (25.5, 3), (58, 5), (6.8, 0.4), (158, 25)),
}

# Typical well-managed yield under ideal conditions, kg/ha (rough national-average order of
# magnitude; used for the synthetic yield generator only).
REFERENCE_YIELD_KG_HA = {
    "rice": 4000, "maize": 3200, "chickpea": 1100, "kidneybeans": 1500, "pigeonpeas": 900,
    "mothbeans": 500, "mungbean": 650, "blackgram": 650, "lentil": 1000, "pomegranate": 12000,
    "banana": 35000, "mango": 8000, "grapes": 20000, "watermelon": 25000, "muskmelon": 18000,
    "apple": 9000, "orange": 12000, "papaya": 40000, "coconut": 7000, "cotton": 1800,
    "jute": 2500, "coffee": 900,
}

# Pseudo agro-climatic regions for the synthetic yield data: (temp shift °C, rain multiplier,
# yield multiplier). Only used to demonstrate leave-one-region-out evaluation.
REGIONS = {
    "north": (-2.0, 0.85, 0.95),
    "south": (2.5, 1.20, 1.05),
    "east": (1.0, 1.30, 1.00),
    "west": (1.5, 0.80, 0.92),
    "central": (0.0, 1.00, 1.08),
}

# Short agronomic advice shown next to out-of-range inputs. General guidance only.
ADVICE = {
    "N": {
        "low": "Nitrogen is below the usual range for {crop}: consider split urea doses or organic manure/compost.",
        "high": "Nitrogen is above the usual range for {crop}: skip N top-dressing to avoid lodging, pests and leaching.",
    },
    "P": {
        "low": "Phosphorus is low for {crop}: a basal dose of DAP / single super phosphate would help.",
        "high": "Phosphorus is already high for {crop}: skip P fertiliser this season.",
    },
    "K": {
        "low": "Potassium is low for {crop}: apply muriate of potash (MOP) or wood ash / compost.",
        "high": "Potassium is already high for {crop}: no extra K needed.",
    },
    "ph": {
        "low": "Soil is more acidic than {crop} prefers: lime (or dolomite) and organic matter can raise pH.",
        "high": "Soil is more alkaline than {crop} prefers: gypsum / elemental sulphur and organic matter can lower pH.",
    },
    "temperature": {
        "low": "Recent temperatures are cooler than {crop} likes: consider delaying sowing or protected cultivation.",
        "high": "Recent temperatures are hotter than {crop} likes: use mulching and consider shifting the sowing window.",
    },
    "humidity": {
        "low": "Air is drier than {crop} prefers: plan irrigation and mulching to protect moisture.",
        "high": "Air is more humid than {crop} prefers: watch for fungal disease and keep good spacing / airflow.",
    },
    "rainfall": {
        "low": "Rainfall is below what {crop} typically gets: plan supplementary irrigation.",
        "high": "Rainfall is above what {crop} typically gets: make sure the field drains well.",
    },
}
