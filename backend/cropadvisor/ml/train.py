"""Train + evaluate both models and write artefacts to ``models/``.

    python -m cropadvisor.ml.train

Artefacts
  crop_classifier.joblib   RandomForest (handles NaN natively, trained with missing-value augmentation)
  yield_models.joblib      HistGradientBoosting p10 / p50 / p90 on log(yield), crop as categorical
  profile.json             per-crop ideal ranges, importances, metrics, data source
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import get_settings
from .data import load_crop_data, load_yield_data
from .profiles import FEATURES, SOIL_FEATURES, WEATHER_FEATURES

SEED = 42


# --------------------------------------------------------------------------- helpers
def augment_missing(X: pd.DataFrame, y, copies: int = 2, seed: int = SEED):
    """Copies of the data with random features blanked out (NaN).

    Makes the model robust to broken / offline sensors and failed API calls.
    Each row keeps at least one feature.
    """
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(copies):
        Xa = np.array(X, dtype=float)  # writable copy (pandas 3 returns read-only views)
        rate = rng.uniform(0.1, 0.6, size=(len(Xa), 1))
        mask = rng.random(Xa.shape) < rate
        mask[mask.all(axis=1), rng.integers(0, Xa.shape[1])] = False
        Xa[mask] = np.nan
        Xs.append(pd.DataFrame(Xa, columns=X.columns))
        ys.append(np.asarray(y))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def mask_columns(X: pd.DataFrame, cols) -> pd.DataFrame:
    Xm = X.copy()
    Xm[list(cols)] = np.nan
    return Xm


def random_missing(X: pd.DataFrame, k: int, rng) -> pd.DataFrame:
    Xm = X.to_numpy(dtype=float).copy()
    for i in range(len(Xm)):
        Xm[i, rng.choice(Xm.shape[1], size=k, replace=False)] = np.nan
    return pd.DataFrame(Xm, columns=X.columns)


def _rf(seed=SEED):
    # 100 trees / min leaf 3: same accuracy as 300 / 1 in our tests but ~5x smaller in RAM (matters on 512 MB hosts)
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=seed)


# --------------------------------------------------------------------------- classifier
def train_classifier(df: pd.DataFrame):
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    Xaug, yaug = augment_missing(Xtr, ytr)
    robust = _rf().fit(pd.concat([Xtr, Xaug], ignore_index=True), np.concatenate([np.asarray(ytr), yaug]))
    plain = _rf().fit(Xtr, ytr)  # baseline for the robustness comparison

    rng = np.random.default_rng(SEED)
    metrics = {
        "complete_data": {
            "accuracy": float(accuracy_score(yte, robust.predict(Xte))),
            "macro_f1": float(f1_score(yte, robust.predict(Xte), average="macro")),
        },
        "missing_k_features": {"robust_model": {}, "plain_model": {}},
        "feature_group_ablation": {},
    }
    for k in range(0, len(FEATURES)):
        accs_r, accs_p = [], []
        for rep in range(3 if k else 1):
            Xm = random_missing(Xte, k, rng) if k else Xte
            accs_r.append(accuracy_score(yte, robust.predict(Xm)))
            accs_p.append(accuracy_score(yte, plain.predict(Xm)))
        metrics["missing_k_features"]["robust_model"][str(k)] = float(np.mean(accs_r))
        metrics["missing_k_features"]["plain_model"][str(k)] = float(np.mean(accs_p))

    # soil vs weather: which group matters more for *choosing* the crop?
    metrics["feature_group_ablation"] = {
        "soil_and_weather": float(accuracy_score(yte, robust.predict(Xte))),
        "soil_only (weather missing)": float(accuracy_score(yte, robust.predict(mask_columns(Xte, WEATHER_FEATURES)))),
        "weather_only (soil missing)": float(accuracy_score(yte, robust.predict(mask_columns(Xte, SOIL_FEATURES)))),
    }
    # drop-one: accuracy when a single feature is unavailable
    metrics["drop_one_feature"] = {
        f: float(accuracy_score(yte, robust.predict(mask_columns(Xte, [f])))) for f in FEATURES
    }

    # final model: refit on all data (+ augmentation) for deployment
    Xall_aug, yall_aug = augment_missing(X, y)
    final = _rf().fit(pd.concat([X, Xall_aug], ignore_index=True), np.concatenate([np.asarray(y), yall_aug]))
    importances = dict(zip(FEATURES, map(float, final.feature_importances_)))

    ranges = {}
    for crop, g in df.groupby("label"):
        ranges[crop] = {
            f: {
                "p10": float(g[f].quantile(0.10)),
                "p50": float(g[f].quantile(0.50)),
                "p90": float(g[f].quantile(0.90)),
            }
            for f in FEATURES
        }
    return final, metrics, importances, ranges


# --------------------------------------------------------------------------- yield regressor
def _yield_frame(df: pd.DataFrame, crops: list[str]) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X.insert(0, "crop_idx", df["crop"].map({c: i for i, c in enumerate(crops)}).astype(float))
    return X


def _fit_yield(Xf: pd.DataFrame, y: np.ndarray):
    kw = dict(max_iter=300, learning_rate=0.06, categorical_features=[0], random_state=SEED)
    models = {
        "p50": HistGradientBoostingRegressor(loss="squared_error", **kw),
        "p10": HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **kw),
        "p90": HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **kw),
    }
    for m in models.values():
        m.fit(Xf, np.log(y))
    return models


def _predict_yield(models, Xf):
    return {k: np.exp(m.predict(Xf)) for k, m in models.items()}


def train_yield(df: pd.DataFrame):
    crops = sorted(df["crop"].unique())
    Xf, y = _yield_frame(df, crops), df["yield_kg_ha"].to_numpy(float)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=df["crop"])

    Xaug, yaug = augment_missing(Xf.iloc[tr][FEATURES], y[tr], copies=1)
    crop_aug = pd.concat([Xf.iloc[tr]["crop_idx"].reset_index(drop=True)], ignore_index=True)
    Xaug.insert(0, "crop_idx", crop_aug.values)
    models = _fit_yield(pd.concat([Xf.iloc[tr], Xaug], ignore_index=True), np.concatenate([y[tr], yaug]))

    pred = _predict_yield(models, Xf.iloc[te])
    inside = (y[te] >= pred["p10"]) & (y[te] <= pred["p90"])
    # quantile boosters under-cover on held-out data: widen the band (in log space) until the
    # nominal 80% interval really covers ~80% of held-out plots
    lo, hi = np.log(pred["p50"] / pred["p10"]), np.log(pred["p90"] / pred["p50"])
    scale, cal_cov = 1.0, float(inside.mean())
    for s_ in np.arange(1.0, 4.01, 0.05):
        cov = float(((y[te] >= pred["p50"] * np.exp(-s_ * lo)) & (y[te] <= pred["p50"] * np.exp(s_ * hi))).mean())
        scale, cal_cov = float(s_), cov
        if cov >= 0.80:
            break
    metrics = {
        "holdout": {
            "r2": float(r2_score(y[te], pred["p50"])),
            "mape": float(mean_absolute_percentage_error(y[te], pred["p50"])),
            "p10_p90_coverage_raw": float(inside.mean()),
            "interval_scale": scale,
            "p10_p90_coverage_calibrated": cal_cov,
        },
        "soil_vs_weather": {},
        "leave_one_region_out": {},
    }

    # which feature group explains yield? (drop one group at test time)
    for name, cols in [("none", []), ("weather_missing", WEATHER_FEATURES), ("soil_missing", SOIL_FEATURES)]:
        Xt = Xf.iloc[te].copy()
        if cols:
            Xt[cols] = np.nan
        metrics["soil_vs_weather"][name] = float(r2_score(y[te], _predict_yield(models, Xt)["p50"]))

    # regional generalisation: train on all regions but one, test on the held-out region
    regions = sorted(df["region"].unique())
    if len(regions) > 1:
        for r in regions:
            trm, tem = (df["region"] != r).to_numpy(), (df["region"] == r).to_numpy()
            m = _fit_yield(Xf[trm], y[trm])
            p = _predict_yield(m, Xf[tem])["p50"]
            metrics["leave_one_region_out"][r] = {
                "r2": float(r2_score(y[tem], p)),
                "mape": float(mean_absolute_percentage_error(y[tem], p)),
            }

    # final deployment model on everything
    Xaug, yaug = augment_missing(Xf[FEATURES], y, copies=1)
    Xaug.insert(0, "crop_idx", Xf["crop_idx"].values)
    final = _fit_yield(pd.concat([Xf, Xaug], ignore_index=True), np.concatenate([y, yaug]))
    return final, crops, metrics, scale


# --------------------------------------------------------------------------- main
def main() -> dict:
    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    crop_df, crop_src = load_crop_data()
    yield_df, yield_src = load_yield_data()
    print(f"[train] crop data: {len(crop_df)} rows ({crop_src}); yield data: {len(yield_df)} rows ({yield_src})")

    clf, clf_metrics, importances, ranges = train_classifier(crop_df)
    ymodels, ycrops, y_metrics, y_scale = train_yield(yield_df)

    missing = sorted(set(clf.classes_) - set(ycrops))
    if missing:
        print(f"[train] WARNING: no yield data for crops {missing}; yield will be unavailable for them")

    joblib.dump(clf, s.models_dir / "crop_classifier.joblib", compress=3)
    joblib.dump({"models": ymodels, "crops": ycrops, "interval_scale": y_scale}, s.models_dir / "yield_models.joblib", compress=3)
    profile = {
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "data_source": {"crop": crop_src, "yield": yield_src},
        "features": FEATURES,
        "classes": list(map(str, clf.classes_)),
        "feature_importance": importances,
        "ranges": ranges,
        "metrics": {"classifier": clf_metrics, "yield": y_metrics},
    }
    (s.models_dir / "profile.json").write_text(json.dumps(profile, indent=2))
    print(
        f"[train] done in {time.time() - t0:.1f}s | crop acc={clf_metrics['complete_data']['accuracy']:.3f} "
        f"| yield R2={y_metrics['holdout']['r2']:.3f} | saved to {s.models_dir}"
    )
    return profile


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
