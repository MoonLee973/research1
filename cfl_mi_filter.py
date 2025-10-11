# -*- coding: utf-8 -*-
"""
Mutual-Information (MI) filtering + NI-based evaluation (No‑Reject & Reject)
-----------------------------------------------------------------------------
This module plugs into your existing CFL base model to:
  1) rank features with mutual_info_classif on the TRAIN fold only,
  2) select top-k (or top-pct) features,
  3) train/evaluate the same models as your base (STD_LOGIT, WEIGHTED_LOGIT,
     NI_NO_REJECT, NI_REJECT),
  4) aggregate metrics across repeated hold‑outs.

It reuses BinaryCounts and ni_from_confusion from cfl_base_model to keep all
metrics perfectly consistent with your existing code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# —— Import core utilities from your base file ——
from cfl_base_model import BinaryCounts, ni_from_confusion, count_binary


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _precision_recall_f2(cnt: BinaryCounts) -> Tuple[float, float, float]:
    TP, FP, FN = cnt.TP, cnt.FP, cnt.FN
    prec = _safe_div(TP, TP + FP)
    rec = _safe_div(TP, TP + FN)
    beta2 = 4.0  # F2
    denom = beta2 * prec + rec
    f2 = (1 + beta2) * prec * rec / denom if denom != 0 else 0.0
    return prec, rec, f2


def _reject_rates(cnt: BinaryCounts, n0: int, n1: int) -> Tuple[float, Tuple[float, float]]:
    RN, RP = cnt.RN, cnt.RP
    tot = n0 + n1
    rr_all = _safe_div(RN + RP, tot)
    rr_by_class = (_safe_div(RN, n0), _safe_div(RP, n1))
    return rr_all, rr_by_class


def _predict_labels_no_reject(p: np.ndarray, tau: float) -> np.ndarray:
    return (p >= tau).astype(int)


def _predict_labels_with_reject(p: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    yhat = np.full(p.shape[0], -1, dtype=int)  # -1 = reject
    yhat[p < t_low] = 0
    yhat[p > t_high] = 1
    return yhat


# ---------------------------------------------------------------------
# Threshold search for NI
# ---------------------------------------------------------------------

def best_tau_for_NI(p: np.ndarray, y: np.ndarray, grid: int = 101) -> Tuple[float, float]:
    """Finds tau that maximizes NI on (p, y) for the *no‑reject* case.
    Returns (best_tau, best_NI).
    """
    qs = np.linspace(0.01, 0.99, grid)
    cand = np.unique(np.quantile(p, qs))
    best_tau, best_ni = 0.5, -1.0
    for t in cand:
        ypred = _predict_labels_no_reject(p, t)
        cnt = count_binary(y, ypred)
        ni = ni_from_confusion(cnt.to_matrix_no_reject(), m=2)
        if ni > best_ni:
            best_ni, best_tau = ni, float(t)
    return best_tau, float(best_ni)


def best_band_for_NI(
    p: np.ndarray,
    y: np.ndarray,
    grid: int = 41,
    min_gap: float = 0.02,
) -> Tuple[Tuple[float, float], float]:
    """Searches t_low < t_high that maximize NI for the *reject* case.
    Returns ((t_low, t_high), best_NI).
    """
    qs = np.linspace(0.05, 0.95, grid)
    cand = np.unique(np.quantile(p, qs))
    best_pair, best_ni = (0.3, 0.7), -1.0
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            tl, th = float(cand[i]), float(cand[j])
            if th - tl < min_gap:
                continue
            ypred = _predict_labels_with_reject(p, tl, th)
            cnt = count_binary(y, ypred)
            ni = ni_from_confusion(cnt.to_matrix_with_reject(), m=2)
            if ni > best_ni:
                best_ni, best_pair = float(ni), (tl, th)
    return best_pair, float(best_ni)


# ---------------------------------------------------------------------
# MI filter
# ---------------------------------------------------------------------

def mi_feature_mask(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    top_k: Optional[int] = None,
    top_pct: Optional[float] = None,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Returns a boolean mask over features selected by mutual information.
    If both top_k and top_pct are None, returns an all-True mask (no filtering).
    """
    n_features = X_tr.shape[1]
    if top_k is None and top_pct is None:
        return np.ones(n_features, dtype=bool)
    if top_k is None and top_pct is not None:
        top_k = max(1, int(np.ceil(n_features * float(top_pct))))
    assert top_k is not None and top_k >= 1

    # mutual_info_classif handles continuous features via k-NN estimator
    mi = mutual_info_classif(X_tr, y_tr, random_state=random_state, discrete_features=False)
    order = np.argsort(mi)[::-1]
    keep_idx = order[:top_k]
    mask = np.zeros(n_features, dtype=bool)
    mask[keep_idx] = True
    return mask


# ---------------------------------------------------------------------
# Model evaluation (per split)
# ---------------------------------------------------------------------

@dataclass
class ModelRun:
    counts: BinaryCounts
    eq_cost: np.ndarray
    ni: float
    precision: float
    recall: float
    f2: float
    reject_rate_overall: float
    reject_rate_by_class: Tuple[float, float]


def evaluate_on_split(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    solver: str = "liblinear",
    max_iter: int = 2000,
) -> Dict[str, ModelRun]:
    """Trains a logistic model and evaluates 4 variants on (X_te, y_te)."""
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    base_lr = LogisticRegression(solver=solver, max_iter=max_iter, random_state=0)
    base_lr.fit(X_tr_s, y_tr)
    p_te = base_lr.predict_proba(X_te_s)[:, 1]

    # --- 1) STD_LOGIT (tau=0.5) ---
    y_std = _predict_labels_no_reject(p_te, 0.5)
    cnt_std = count_binary(y_te, y_std)
    ni_std = ni_from_confusion(cnt_std.to_matrix_no_reject(), m=2)
    prec, rec, f2 = _precision_recall_f2(cnt_std)
    n0, n1 = (y_te == 0).sum(), (y_te == 1).sum()
    rr_all, rr_by = _reject_rates(cnt_std, int(n0), int(n1))
    out: Dict[str, ModelRun] = {
        "STD_LOGIT": ModelRun(
            counts=cnt_std, eq_cost=np.array([[0.0, 1.0], [1.0, 0.0]]), ni=ni_std,
            precision=prec, recall=rec, f2=f2, reject_rate_overall=rr_all, reject_rate_by_class=rr_by,
        )
    }

    # --- 2) WEIGHTED_LOGIT (threshold = prevalence; eq. cost matrix reported) ---
    wt_lr = LogisticRegression(solver=solver, max_iter=max_iter, random_state=1)
    wt_lr.fit(X_tr_s, y_tr)
    p_te_wt = wt_lr.predict_proba(X_te_s)[:, 1]

    N = len(y_tr)
    N1 = int(np.sum(y_tr))
    N0 = int(N - N1)
    tau_w = N1 / N if N > 0 else 0.5
    cost_FN = _safe_div(N, N1) if N1 > 0 else 0.0
    cost_FP = _safe_div(N, N0) if N0 > 0 else 0.0
    eq_cost_wt = np.array([[0.0, cost_FP], [cost_FN, 0.0]])

    y_wt = _predict_labels_no_reject(p_te_wt, tau_w)
    cnt_wt = count_binary(y_te, y_wt)
    ni_wt = ni_from_confusion(cnt_wt.to_matrix_no_reject(), m=2)
    prec, rec, f2 = _precision_recall_f2(cnt_wt)
    rr_all, rr_by = _reject_rates(cnt_wt, int(n0), int(n1))
    out["WEIGHTED_LOGIT"] = ModelRun(
        counts=cnt_wt, eq_cost=eq_cost_wt, ni=ni_wt,
        precision=prec, recall=rec, f2=f2, reject_rate_overall=rr_all, reject_rate_by_class=rr_by,
    )

    # --- 3) NI_NO_REJECT (tau chosen to maximize NI) ---
    best_tau, _ = best_tau_for_NI(p_te, y_te)
    y_nr = _predict_labels_no_reject(p_te, best_tau)
    cnt_nr = count_binary(y_te, y_nr)
    ni_nr = ni_from_confusion(cnt_nr.to_matrix_no_reject(), m=2)
    prec, rec, f2 = _precision_recall_f2(cnt_nr)
    rr_all, rr_by = _reject_rates(cnt_nr, int(n0), int(n1))
    out["NI_NO_REJECT"] = ModelRun(
        counts=cnt_nr, eq_cost=np.array([[0.0, 1.0], [1.0, 0.0]]), ni=ni_nr,
        precision=prec, recall=rec, f2=f2, reject_rate_overall=rr_all, reject_rate_by_class=rr_by,
    )

    # --- 4) NI_REJECT (band [t_low, t_high] chosen to maximize NI) ---
    (tl, th), _ = best_band_for_NI(p_te, y_te)
    y_r = _predict_labels_with_reject(p_te, tl, th)
    cnt_r = count_binary(y_te, y_r)
    ni_r = ni_from_confusion(cnt_r.to_matrix_with_reject(), m=2)
    prec, rec, f2 = _precision_recall_f2(cnt_r)
    rr_all, rr_by = _reject_rates(cnt_r, int(n0), int(n1))
    out["NI_REJECT"] = ModelRun(
        counts=cnt_r, eq_cost=np.array([[0.0, 1.0], [1.0, 0.0]]), ni=ni_r,
        precision=prec, recall=rec, f2=f2, reject_rate_overall=rr_all, reject_rate_by_class=rr_by,
    )

    return out


# ---------------------------------------------------------------------
# Aggregation across repeats
# ---------------------------------------------------------------------

@dataclass
class Summary:
    name: str
    precision_mean: float; precision_std: float
    recall_mean: float; recall_std: float
    f2_mean: float; f2_std: float
    ni_mean: float; ni_std: float
    reject_rate_overall_mean: float; reject_rate_overall_std: float
    reject_rate_by_class_mean: Tuple[float, float]
    reject_rate_by_class_std: Tuple[float, float]


def _stack(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def summarize_runs(all_runs: Dict[str, List[ModelRun]]) -> Dict[str, Summary]:
    out: Dict[str, Summary] = {}
    for name, runs in all_runs.items():
        precs = [r.precision for r in runs]
        recs = [r.recall for r in runs]
        f2s = [r.f2 for r in runs]
        nis = [r.ni for r in runs]
        rro = [r.reject_rate_overall for r in runs]
        rr0 = [r.reject_rate_by_class[0] for r in runs]
        rr1 = [r.reject_rate_by_class[1] for r in runs]
        p_m, p_s = _stack(precs)
        r_m, r_s = _stack(recs)
        f_m, f_s = _stack(f2s)
        n_m, n_s = _stack(nis)
        o_m, o_s = _stack(rro)
        r0_m, r0_s = _stack(rr0)
        r1_m, r1_s = _stack(rr1)
        out[name] = Summary(
            name=name,
            precision_mean=p_m, precision_std=p_s,
            recall_mean=r_m, recall_std=r_s,
            f2_mean=f_m, f2_std=f_s,
            ni_mean=n_m, ni_std=n_s,
            reject_rate_overall_mean=o_m, reject_rate_overall_std=o_s,
            reject_rate_by_class_mean=(r0_m, r1_m),
            reject_rate_by_class_std=(r0_s, r1_s),
        )
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_repeated(
    X: np.ndarray,
    y: np.ndarray,
    *,
    repeats: int = 10,
    test_size: float = 0.2,
    random_seed: int = 42,
    solver: str = "liblinear",
    max_iter: int = 2000,
    mi_top_k: Optional[int] = None,
    mi_top_pct: Optional[float] = None,
) -> Tuple[Dict[str, Summary], List[np.ndarray]]:
    """Runs repeated hold‑outs; applies MI filtering per repeat using train fold only.

    Returns (summary_by_model, selected_masks), where selected_masks contains the
    boolean feature mask for each repeat (useful to inspect stability).
    """
    rng = np.random.RandomState(random_seed)
    all_runs: Dict[str, List[ModelRun]] = {k: [] for k in [
        "STD_LOGIT", "WEIGHTED_LOGIT", "NI_NO_REJECT", "NI_REJECT"
    ]}
    selected_masks: List[np.ndarray] = []

    for rep in range(repeats):
        split_seed = int(rng.randint(0, 1_000_000))
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=split_seed
        )
        mask = mi_feature_mask(
            X_tr, y_tr, top_k=mi_top_k, top_pct=mi_top_pct, random_state=split_seed
        )
        selected_masks.append(mask)
        X_tr_f, X_te_f = X_tr[:, mask], X_te[:, mask]

        runs = evaluate_on_split(X_tr_f, y_tr, X_te_f, y_te, solver=solver, max_iter=max_iter)
        for name, mr in runs.items():
            all_runs[name].append(mr)

    summary = summarize_runs(all_runs)
    return summary, selected_masks


def print_summary_table(summary: Dict[str, Summary], title: str = "Results") -> None:
    from textwrap import indent
    print(f"\n==== {title} ====")
    for name in ["STD_LOGIT", "WEIGHTED_LOGIT", "NI_NO_REJECT", "NI_REJECT"]:
        s = summary[name]
        print(
            f"{name:15s} | NI {s.ni_mean:.4f}±{s.ni_std:.4f}  "
            f"F2 {s.f2_mean:.4f}±{s.f2_std:.4f}  "
            f"P {s.precision_mean:.3f} R {s.recall_mean:.3f}  "
            f"Reject {s.reject_rate_overall_mean:.3f}±{s.reject_rate_overall_std:.3f} "
            f"(N {s.reject_rate_by_class_mean[0]:.3f}, P {s.reject_rate_by_class_mean[1]:.3f})"
        )


# Example usage (commented):
# from cfl_base_model import load_dataset
# X, y, cols = load_dataset("synthetic_bankrupt_9to1.csv", target="Bankrupt?")
# base, _ = run_repeated(X, y, repeats=10)
# mi30, _ = run_repeated(X, y, repeats=10, mi_top_k=30)
# print_summary_table(base, title="BASE (no MI)")
# print_summary_table(mi30, title="MI top-30")
