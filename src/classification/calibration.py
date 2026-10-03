"""Per-class probability calibration (C5): temperature, Platt and isotonic, plus calibration metrics.

Calibration maps a raw sigmoid score to a calibrated probability. It never changes the C4 binary
decisions: those keep using the raw score with the C4 threshold (or, equivalently, the calibrated
score with the calibrated-equivalent threshold computed here).

Conventions
  * z = logit(p) = log(p / (1 - p)), computed in float64 with a documented clip.
  * Temperature: p_cal = sigmoid(z / T), T > 0 (fitted on log T, bounded).
  * Platt:       p_cal = sigmoid(a z + b); the slope must be > 0 or the fit is flagged pathological.
  * Isotonic:    non-decreasing piecewise-linear map of z; outputs clipped to [ISO_EPS, 1 - ISO_EPS].
  * Calibrators are fitted on VALID (non-uncertain) entries of one class at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression

from src.classification.labels import LABELS, assert_matches
from src.data.splits import stratified_group_split

LOGIT_EPS = 1e-12            # clip for logit(p); the C2-B scores never reach it (checked by the runner)
ISO_EPS = 1e-6               # isotonic outputs are clipped so log loss stays finite
T_BOUNDS = (0.05, 20.0)      # temperature search range
GUARD = {"temp_edge_tol": 0.01, "platt_slope_min": 0.1, "platt_slope_max": 3.0, "platt_abs_intercept_max": 5.0,
         "iso_min_levels": 5}
METHODS = ("raw", "temperature", "platt", "isotonic")
COMPLEXITY = {m: i for i, m in enumerate(METHODS)}


class CalibrationError(ValueError):
    pass


# ------------------------------------------------------------------ logit / basics
def check_probs(p: np.ndarray, name: str = "scores") -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise CalibrationError(f"{name} contains NaN or Inf")
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise CalibrationError(f"{name} outside [0, 1]")
    return p


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(check_probs(p), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _check_fit_inputs(z: np.ndarray, y: np.ndarray, min_each: int = 2) -> tuple[np.ndarray, np.ndarray]:
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y).astype(bool)
    if z.shape != y.shape or z.ndim != 1:
        raise CalibrationError("z and y must be 1-D arrays of equal length")
    if not np.all(np.isfinite(z)):
        raise CalibrationError("logits contain NaN or Inf")
    if y.sum() < min_each or (~y).sum() < min_each:
        raise CalibrationError(f"degenerate labels: {int(y.sum())} positives, {int((~y).sum())} negatives")
    return z, y


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    y = np.asarray(y).astype(bool)
    return float(-(np.log(p[y]).sum() + np.log1p(-p[~y]).sum()) / len(y))


# ------------------------------------------------------------------ fitting
def fit_temperature(z: np.ndarray, y: np.ndarray) -> dict:
    z, y = _check_fit_inputs(z, y)
    f = lambda lt: nll(sigmoid(z / np.exp(lt)), y)  # noqa: E731
    r = minimize_scalar(f, bounds=(np.log(T_BOUNDS[0]), np.log(T_BOUNDS[1])), method="bounded", options={"xatol": 1e-10})
    T = float(np.exp(r.x))
    lo, hi = T_BOUNDS
    at_edge = bool(T <= lo * (1 + GUARD["temp_edge_tol"]) or T >= hi * (1 - GUARD["temp_edge_tol"]))
    flags = (["optimizer_failed"] if not r.success else []) + (["T_at_search_boundary"] if at_edge else [])
    return {"method": "temperature", "T": T, "flags": flags, "pathological": bool(flags), "train_nll": float(r.fun)}


def fit_platt(z: np.ndarray, y: np.ndarray) -> dict:
    """Logistic regression y ~ sigmoid(a z + b) by damped Newton (convex problem)."""
    z, y = _check_fit_inputs(z, y)
    yf = y.astype(np.float64)
    a, b = 1.0, 0.0
    converged = False
    cur = nll(sigmoid(a * z + b), y)
    for _ in range(200):
        p = sigmoid(a * z + b)
        g = np.array([np.sum((p - yf) * z), np.sum(p - yf)]) / len(z)
        w = p * (1 - p)
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) / len(z) + 1e-12 * np.eye(2)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-8:  # backtracking line search
            new = nll(sigmoid((a - t * step[0]) * z + (b - t * step[1])), y)
            if new <= cur + 1e-14:
                break
            t /= 2
        a, b = a - t * step[0], b - t * step[1]
        cur, prev = new if t > 1e-8 else cur, cur
        if np.max(np.abs(g)) < 1e-9:
            converged = True
            break
    flags = []
    if not converged:
        flags.append("not_converged")
    if a <= 0:
        flags.append("non_positive_slope")
    elif a < GUARD["platt_slope_min"] or a > GUARD["platt_slope_max"]:
        flags.append("extreme_slope")
    if abs(b) > GUARD["platt_abs_intercept_max"]:
        flags.append("extreme_intercept")
    return {"method": "platt", "a": float(a), "b": float(b), "flags": flags, "pathological": bool(flags), "train_nll": float(cur)}


def fit_isotonic(z: np.ndarray, y: np.ndarray) -> dict:
    z, y = _check_fit_inputs(z, y)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(z, y.astype(float))
    xs, ys = iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)
    levels = int(len(np.unique(np.round(ys, 12))))
    flags = ["very_few_output_levels"] if levels < GUARD["iso_min_levels"] else []
    return {"method": "isotonic", "x": xs.tolist(), "y": ys.tolist(), "n_levels": levels, "n_knots": int(len(xs)),
            "flags": flags, "pathological": bool(flags)}


FITTERS = {"temperature": fit_temperature, "platt": fit_platt, "isotonic": fit_isotonic}


def fit_calibrator(method: str, z: np.ndarray, y: np.ndarray) -> dict:
    if method == "raw":
        return {"method": "raw", "flags": [], "pathological": False}
    if method not in FITTERS:
        raise CalibrationError(f"unknown calibration method {method!r}")
    return FITTERS[method](z, y)


# ------------------------------------------------------------------ application
def apply_calibrator(spec: dict, z: np.ndarray) -> np.ndarray:
    """Calibrated probability for logits z. Output is always inside [0, 1]."""
    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise CalibrationError("logits contain NaN or Inf")
    m = spec["method"]
    if m == "raw":
        return sigmoid(z)
    if m == "temperature":
        if not spec["T"] > 0:
            raise CalibrationError("temperature must be positive")
        return sigmoid(z / spec["T"])
    if m == "platt":
        return sigmoid(spec["a"] * z + spec["b"])
    if m == "isotonic":
        return np.clip(np.interp(z, spec["x"], spec["y"]), ISO_EPS, 1 - ISO_EPS)
    raise CalibrationError(f"unknown calibration method {m!r}")


def is_strictly_increasing(spec: dict) -> bool:
    m = spec["method"]
    if m in ("raw", "temperature"):
        return True
    if m == "platt":
        return spec["a"] > 0
    return False  # isotonic is only non-decreasing (plateaus)


# ------------------------------------------------------------------ metrics
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return nll(np.asarray(p, dtype=float), y)


def adaptive_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-frequency bins (rank-based; ties broken by a stable sort). Empty bins are never produced."""
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(bool)
    n_bins = max(1, min(n_bins, len(p)))
    order = np.argsort(p, kind="mergesort")
    out = []
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        out.append({"n": int(len(idx)), "mean_predicted": float(p[idx].mean()), "observed_fraction": float(y[idx].mean()),
                    "n_positive": int(y[idx].sum())})
    return out


def ece_adaptive(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = adaptive_bins(p, y, n_bins)
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] / n * abs(b["mean_predicted"] - b["observed_fraction"]) for b in bins))


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray) -> dict:
    """Slope: logistic regression of y on logit(p). Intercept (calibration-in-the-large): offset b with slope fixed at 1."""
    z, yb = safe_logit(p), np.asarray(y).astype(bool)
    if yb.sum() < 2 or (~yb).sum() < 2:
        return {"cal_slope": float("nan"), "cal_intercept_slope1": float("nan")}
    spec = fit_platt(z, yb)
    yf = yb.astype(float)
    r = minimize_scalar(lambda b: nll(sigmoid(z + b), yb), bounds=(-10, 10), method="bounded", options={"xatol": 1e-10})
    return {"cal_slope": spec["a"], "cal_intercept_slope1": float(r.x)}


def class_metrics(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    p = check_probs(p)
    return {"brier": brier(p, y), "log_loss": log_loss(p, y), "ece": ece_adaptive(p, y, n_bins),
            "mean_predicted": float(p.mean()), "prevalence": float(np.asarray(y).astype(bool).mean()),
            **calibration_slope_intercept(p, y)}


# ------------------------------------------------------------------ patient-level folds
def patient_folds(patient_ids, valid: np.ndarray, raw_labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index per image (patients never split). Multi-label iterative stratification over patients,
    stratified on the number of VALID POSITIVE images per class."""
    pat = np.asarray(patient_ids)
    uniq, inv = np.unique(pat, return_inverse=True)
    pos = ((raw_labels == 1) & valid).astype(float)
    counts = np.zeros((len(uniq), pos.shape[1]))
    np.add.at(counts, inv, pos)
    sizes = np.bincount(inv)
    names = [f"fold{k}" for k in range(n_folds)]
    assign = stratified_group_split(uniq, counts, sizes, {n: 1 / n_folds for n in names}, seed)
    fold_of_patient = np.array([names.index(assign[str(u)]) for u in uniq])
    folds = fold_of_patient[inv]
    # leakage guard
    for k in range(n_folds):
        for k2 in range(k + 1, n_folds):
            if set(pat[folds == k]) & set(pat[folds == k2]):
                raise CalibrationError("patient appears in two folds")
    return folds


# ------------------------------------------------------------------ threshold mapping
def calibrated_threshold(spec: dict, raw_threshold: float) -> float:
    return float(apply_calibrator(spec, safe_logit(np.array([raw_threshold])))[0])


def decision_mismatches(raw_scores: np.ndarray, raw_threshold: float, spec: dict) -> int:
    """Number of images where (raw >= raw_thr) != (calibrated >= calibrated-equivalent thr)."""
    z = safe_logit(raw_scores)
    cal_thr = calibrated_threshold(spec, raw_threshold)
    return int(((raw_scores >= raw_threshold) != (apply_calibrator(spec, z) >= cal_thr)).sum())


# ------------------------------------------------------------------ files
def save_calibrators(path: Path, specs: dict, meta: dict) -> None:
    for lab in specs:
        if lab not in LABELS:
            raise CalibrationError(f"unknown class {lab!r}")
    payload = {**meta, "label_order": list(LABELS), "classes": {lab: specs[lab] for lab in LABELS if lab in specs}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_calibrators(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_matches(d["label_order"])
    missing = [l for l in LABELS if l not in d["classes"]]
    if missing:
        raise CalibrationError(f"missing calibrators for {missing}")
    for lab, s in d["classes"].items():
        if s["method"] not in METHODS:
            raise CalibrationError(f"{lab}: unknown method {s['method']!r}")
    return d
