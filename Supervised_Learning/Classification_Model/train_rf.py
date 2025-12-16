#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json, argparse, warnings
from pathlib import Path
from typing import List, Union, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    f1_score, accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score,
    precision_score, recall_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import TransformedTargetRegressor

import joblib


# ============== I/O and Preprocessing ==============

def read_csv_robust(path: str) -> pd.DataFrame:
    encs = ["utf-8","utf-8-sig","gbk","cp936","latin1","iso-8859-1","utf-16","utf-16le","utf-16be"]
    seps = [",","\t",";","|"]
    last_err = None
    for e in encs:
        for s in seps:
            try:
                df = pd.read_csv(path, encoding=e, sep=s, low_memory=False)
                if df.shape[1] >= 1:
                    return df
            except Exception as err:
                last_err = err
    if last_err:
        raise last_err
    raise RuntimeError(f"Unable to read CSV: {path}")

def split_xy(df: pd.DataFrame, target: str, drop_cols: List[str]):
    """
    Split features and target:
    - Automatically exclude row_index / index / id / SMILES / RecipeID / SampleID, etc.
    - Keep only numeric feature columns.
    """
    exclude = set(drop_cols + [target])
    auto_exclude = {
        c for c in df.columns
        if str(c).strip().lower() in {
            "row_index", "index", "id", "idx", "sampleid", "recipeid",
            "smile", "smiles", "smile a", "smile b", "smile c"
        }
    }
    exclude |= auto_exclude

    cols = [
        c for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]
    if not cols:
        raise ValueError("No numeric feature columns detected. Please check the input CSV.")

    print(f"[INFO] Using {len(cols)} feature columns for training (Excluded: {sorted(exclude)})")

    X = df[cols].copy()
    y = df[target].values
    return X, y, cols

def dump_pred_table(df_src: pd.DataFrame,
                    y_true: Optional[np.ndarray],
                    y_pred: np.ndarray,
                    id_cols: List[str],
                    out_path: Union[str, Path],
                    extra_cols: Dict[str, Union[int, float, str, np.ndarray]] = None):
    out = pd.DataFrame({"idx": np.arange(len(y_pred)), "y_pred": y_pred})
    if y_true is not None:
        out["y_true"] = y_true
        out["residual"] = out["y_true"] - out["y_pred"] if np.issubdtype(np.array(y_pred).dtype, np.number) else None

    keep = []
    if id_cols:
        for c in id_cols:
            if c in df_src.columns:
                keep.append(c)
    if keep:
        out = pd.concat([df_src.reset_index(drop=True)[keep], out], axis=1)

    if extra_cols:
        for k, v in extra_cols.items():
            out[k] = np.asarray(v) if not np.isscalar(v) else v

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[saved]", out_path, " shape=", out.shape)


# ============== Helper Functions ==============

def parse_max_features(val, default=0.25):
    """Parse rf_max_features: 0<float<=1, int, 'sqrt'/'log2'/'auto'. 'auto' maps to 1.0"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip().lower()
    if s in {"sqrt", "log2"}:
        return s
    if s in {"auto", "all"}:
        warnings.warn("rf_max_features='auto' mapped to 1.0 (using all features).", RuntimeWarning)
        return 1.0
    try:
        if s.isdigit():
            return int(s)
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except Exception as e:
        raise ValueError(f"Unable to parse --rf_max_features={val!r}") from e

def parse_min_samples_split(val, default=2):
    """Allow: int>=2 or float in (0,1]"""
    if val is None:
        return default
    if isinstance(val, (int, np.integer)):
        if val >= 2:
            return int(val)
        raise ValueError("min_samples_split int must be >=2")
    try:
        f = float(val)
    except Exception as e:
        raise ValueError(f"Unable to parse rf_min_samples_split={val!r}") from e
    if 0.0 < f <= 1.0:
        return f
    if f.is_integer() and f >= 2.0:
        return int(f)
    raise ValueError("min_samples_split must be int>=2 or float in (0,1]")

def predict_with_threshold_if_needed(model, X, decision_threshold, classes_):
    """
    If decision_threshold is set and it is a binary classification, output labels based on predict_proba using that threshold;
    Otherwise, return (predict result, None). The second return value is used to save the argmax reference.
    """
    y_pred_argmax = model.predict(X)
    if decision_threshold is None or classes_ is None or len(classes_) != 2:
        return y_pred_argmax, None

    proba = getattr(model, "predict_proba", None)
    if proba is None:
        return y_pred_argmax, None

    P = proba(X)
    pos = max(classes_)
    neg = min(classes_)
    pos_idx = list(classes_).index(pos)
    thr = float(decision_threshold)
    y_pred_thr = np.where(P[:, pos_idx] >= thr, pos, neg)
    return y_pred_thr, y_pred_argmax


# ============== Scoring (Regression & Classification) ==============

def eval_metrics_reg(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

def eval_metrics_cls(y_true, y_pred, y_proba=None, classes_: Optional[np.ndarray]=None) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uniq = np.unique(y_true)

    # Precision/Recall definition: Binary uses 'binary' (pos_label=larger label), multiclass uses 'macro'
    avg = "macro"
    pos_label = None
    if len(uniq) == 2:
        pos_label = uniq.max()
        avg = "binary"

    out = {
        "F1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "BalAcc": float(balanced_accuracy_score(y_true, y_pred)),
        "Acc": float(accuracy_score(y_true, y_pred)),
    }
    try:
        if avg == "binary":
            out["Precision"] = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            out["Recall"]    = float(recall_score(y_true, y_pred,    pos_label=pos_label, zero_division=0))
        else:
            out["Precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            out["Recall_macro"]    = float(recall_score(y_true, y_pred,    average="macro", zero_division=0))
    except Exception:
        pass

    # AUC: Binary -> ROC_AUC; Multiclass -> ROC_AUC_ovr
    if y_proba is not None and len(uniq) > 1:
        try:
            classes = np.array(classes_) if classes_ is not None else np.unique(y_true)
            if len(classes) == 2:
                pos = classes.max()
                pos_idx = list(classes).index(pos)
                out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba[:, pos_idx]))
            else:
                out["ROC_AUC_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            pass
    return out

def per_class_accuracy(y_true, y_pred) -> Dict[str, Optional[float]]:
    """
    Class-wise accuracy: For each appearing class c, calculate Acc_class_{c} = (# y_true==c and pred==c) / (# y_true==c)
    """
    out = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    for c in classes:
        mask = (y_true == c)
        if mask.any():
            out[f"Acc_class_{c}"] = float(np.mean(y_pred[mask] == c))
        else:
            out[f"Acc_class_{c}"] = None
    return out


# ============== RF Construction (Regression & Classification) ==============

def default_rf_reg(seed: int, args=None) -> RandomForestRegressor:
    defaults = dict(
        rf_n_estimators=900,
        rf_max_depth=16,
        rf_max_features=0.25,
        rf_min_samples_leaf=0.01,   # fraction
        rf_min_samples_split=0.02,  # fraction
        rf_max_samples=0.7,         # subsample
        rf_ccp_alpha=0.0,
        rf_min_impurity_decrease=0.0,
        rf_oob=False,
    )
    def get(name):
        return getattr(args, name, None) if (args is not None and hasattr(args, name) and getattr(args, name) is not None) else defaults[name]
    mf  = parse_max_features(get('rf_max_features'), default=defaults['rf_max_features'])
    mss = parse_min_samples_split(get('rf_min_samples_split'), default=defaults['rf_min_samples_split'])
    return RandomForestRegressor(
        n_estimators=get('rf_n_estimators'),
        max_depth=get('rf_max_depth'),
        max_features=mf,
        min_samples_leaf=get('rf_min_samples_leaf'),
        min_samples_split=mss,
        n_jobs=-1,
        random_state=seed,
        bootstrap=True,
        max_samples=get('rf_max_samples'),
        ccp_alpha=get('rf_ccp_alpha'),
        min_impurity_decrease=get('rf_min_impurity_decrease'),
        oob_score=bool(get('rf_oob')),
    )

def default_rf_cls(seed: int, args=None) -> RandomForestClassifier:
    defaults = dict(
        rf_n_estimators=800,
        rf_max_depth=16,
        rf_max_features=0.3,
        rf_min_samples_leaf=0.01,       # fraction
        rf_min_samples_split=0.02,      # fraction
        rf_class_weight="balanced_subsample",
        rf_max_samples=0.7,             # subsample
        rf_ccp_alpha=0.0,
        rf_min_impurity_decrease=0.0,
        rf_oob=False,
    )
    def get(name):
        return getattr(args, name, None) if (args is not None and hasattr(args, name) and getattr(args, name) is not None) else defaults[name]
    mf  = parse_max_features(get('rf_max_features'), default=defaults['rf_max_features'])
    mss = parse_min_samples_split(get('rf_min_samples_split'), default=defaults['rf_min_samples_split'])
    return RandomForestClassifier(
        n_estimators=get('rf_n_estimators'),
        max_depth=get('rf_max_depth'),
        max_features=mf,
        min_samples_leaf=get('rf_min_samples_leaf'),
        min_samples_split=mss,
        class_weight=get('rf_class_weight'),
        n_jobs=-1,
        random_state=seed,
        bootstrap=True,
        max_samples=get('rf_max_samples'),
        ccp_alpha=get('rf_ccp_alpha'),
        min_impurity_decrease=get('rf_min_impurity_decrease'),
        oob_score=bool(get('rf_oob')),
    )

def build_rf(task: str, seed: int, args):
    if task == "reg":
        return default_rf_reg(seed, args=args)
    else:
        return default_rf_cls(seed, args=args)

def wrap_y_transform_if_needed(task: str, base_estimator, use_log1p: bool):
    if task == "reg" and use_log1p:
        return TransformedTargetRegressor(
            regressor=base_estimator,
            func=np.log1p,
            inverse_func=np.expm1
        )
    return base_estimator


# ============== Training/Evaluation (Single Split) ==============

def single_split_train_eval(
    task: str,
    df_feat: pd.DataFrame,
    target: str,
    valid_size: float,
    seed: int,
    drop_cols: List[str],
    save_dir: Path,
    # Others
    y_log1p: bool,
    group_col: str,
    n_splits: int,
    fold_idx: int,
    no_perm: bool,
    # Export predictions
    save_train_pred: bool,
    id_cols: List[str],
    args=None,
):
    use_group = bool(group_col) and (group_col in df_feat.columns)
    if use_group:
        groups = df_feat[group_col].values
        gkf = GroupKFold(n_splits=int(n_splits))
        splits = list(gkf.split(df_feat, groups=groups))
        tr_idx, va_idx = splits[fold_idx % len(splits)]
        df_tr, df_va = df_feat.iloc[tr_idx], df_feat.iloc[va_idx]
    else:
        if task == "cls":
            df_tr, df_va = train_test_split(df_feat, test_size=valid_size, random_state=seed, stratify=df_feat[target])
        else:
            df_tr, df_va = train_test_split(df_feat, test_size=valid_size, random_state=seed)

    Xtr, ytr, feat_cols = split_xy(df_tr, target, drop_cols)
    Xva, yva, _         = split_xy(df_va, target, drop_cols)

    base = build_rf(task=task, seed=seed, args=args)
    model = wrap_y_transform_if_needed(task, base, use_log1p=y_log1p)

    Xtr = Xtr.fillna(0.0); Xva = Xva.fillna(0.0)
    model.fit(Xtr, ytr)

    # -------- Prediction and Scoring --------
    if task == "reg":
        pred_tr = model.predict(Xtr)
        pred_va = model.predict(Xva)
        m_tr = eval_metrics_reg(ytr, pred_tr)
        m_va = eval_metrics_reg(yva, pred_va)
        proba_tr = proba_va = None
        classes_ = None
        yhat_tr = pred_tr
        yhat_va = pred_va
    else:
        # Proba and Classes
        proba_tr = getattr(model, "predict_proba", lambda Z: None)(Xtr)
        proba_va = getattr(model, "predict_proba", lambda Z: None)(Xva)
        classes_ = getattr(model, "classes_", None)

        # Threshold/argmax labels
        yhat_tr_thr, yhat_tr_argmax = predict_with_threshold_if_needed(model, Xtr, getattr(args, "decision_threshold", None), classes_)
        yhat_va_thr, yhat_va_argmax = predict_with_threshold_if_needed(model, Xva, getattr(args, "decision_threshold", None), classes_)

        # If threshold not enabled, yhat_* is argmax
        yhat_tr = yhat_tr_thr if yhat_tr_thr is not None else model.predict(Xtr)
        yhat_va = yhat_va_thr if yhat_va_thr is not None else model.predict(Xva)

        # Metrics
        m_tr = eval_metrics_cls(ytr, yhat_tr, proba_tr, classes_)
        m_va = eval_metrics_cls(yva, yhat_va, proba_va, classes_)
        m_tr.update(per_class_accuracy(ytr, yhat_tr))
        m_va.update(per_class_accuracy(yva, yhat_va))

    # If OOB enabled, append OOB score (RF native only)
    if hasattr(model, "oob_score_"):
        try:
            m_tr["OOB"] = float(model.oob_score_)
        except Exception:
            pass

    # Write metrics.json
    metrics_path = save_dir / "metrics.json"
    meta_extra = {
        "use_group": use_group,
        "group_col": group_col if use_group else None,
        "n_splits": int(n_splits) if use_group else None,
        "fold_idx": int(fold_idx) if use_group else None,
        "y_log1p": bool(y_log1p) if task=="reg" else False,
        "model_name": "rf",
        "task": task,
        "decision_threshold": float(getattr(args, "decision_threshold", None)) if getattr(args, "decision_threshold", None) is not None else None
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"train": m_tr, "valid": m_va, "meta": meta_extra}, f, ensure_ascii=False, indent=2)
    print("[metrics] train:", m_tr)
    print("[metrics] valid:", m_va)
    print("[saved]", metrics_path)

    # Export Feature Importance
    try:
        fi = getattr(model, "feature_importances_", None)
        if fi is None and hasattr(model, "regressor_"):
            fi = getattr(model.regressor_, "feature_importances_", None)
        if fi is not None:
            imp = pd.DataFrame({"feature": feat_cols, "importance": fi})
            imp.sort_values("importance", ascending=False, inplace=True)
            imp.to_csv(save_dir/"feature_importance.csv", index=False, encoding="utf-8-sig")
            print("[saved]", save_dir/"feature_importance.csv")
    except Exception as e:
        print("[warn] feature_importance export failed:", e)

    # Classification Confusion Matrix (Validation set, using yhat_va for scoring)
    if task == "cls":
        try:
            cm = confusion_matrix(yva, yhat_va)
            pd.DataFrame(cm).to_csv(save_dir/"confusion_matrix.csv", index=False, encoding="utf-8-sig")
            print("[saved]", save_dir/"confusion_matrix.csv")
        except Exception as e:
            print("[warn] Confusion matrix export failed:", e)

    # Save model payload (Single split)
    payload = {
        "model": model,
        "features": feat_cols,
        "target": target,
        "drop_cols": drop_cols,
        "y_log1p": bool(y_log1p) if task=="reg" else False,
        "group_col": group_col if use_group else None,
        "task": task,
        "decision_threshold": float(getattr(args, "decision_threshold", None)) if getattr(args, "decision_threshold", None) is not None else None
    }
    joblib.dump(payload, save_dir/"best_model.joblib")
    print("[saved]", save_dir/"best_model.joblib")

    if save_train_pred:
        # Validation predictions CSV
        extra_va = {}
        if task == "cls" and proba_va is not None and classes_ is not None and len(classes_) == 2:
            pos = max(classes_)
            pos_idx = list(classes_).index(pos)
            extra_va["proba_pos"] = proba_va[:, pos_idx]
        if task == "cls":
            # If threshold enabled, export argmax reference column
            _, yhat_va_argmax = predict_with_threshold_if_needed(model, Xva, getattr(args, "decision_threshold", None), classes_)
            if getattr(args, "decision_threshold", None) is not None and yhat_va_argmax is not None:
                extra_va["y_pred_argmax"] = yhat_va_argmax
        dump_pred_table(
            df_src=df_va, y_true=yva, y_pred=yhat_va,
            id_cols=id_cols, out_path=save_dir/"valid_predictions.csv", extra_cols=extra_va
        )

        # Training predictions CSV (diagnose overfitting)
        extra_tr = {}
        if task == "cls" and proba_tr is not None and classes_ is not None and len(classes_) == 2:
            pos = max(classes_)
            pos_idx = list(classes_).index(pos)
            extra_tr["proba_pos"] = proba_tr[:, pos_idx]
        if task == "cls":
            _, yhat_tr_argmax = predict_with_threshold_if_needed(model, Xtr, getattr(args, "decision_threshold", None), classes_)
            if getattr(args, "decision_threshold", None) is not None and yhat_tr_argmax is not None:
                extra_tr["y_pred_argmax"] = yhat_tr_argmax
        dump_pred_table(
            df_src=df_tr, y_true=ytr, y_pred=yhat_tr,
            id_cols=id_cols, out_path=save_dir/"train_predictions.csv", extra_cols=extra_tr
        )


# ============== 10-Fold Cross-Validation (Regression & Classification) ==============

def cv10_evaluate_any(
    task: str,
    df_feat: pd.DataFrame,
    target: str,
    seed: int,
    drop_cols: List[str],
    save_dir: Path,
    # Transform (Reg only)
    y_log1p: bool,
    # Grouping
    group_col: str,
    # Export prediction
    save_train_pred: bool,
    id_cols: List[str],
    oof_path: Optional[str] = None,
    args=None,
    # New: Save models for each fold
    save_fold_models: bool = False,
):
    Xall, yall, feat_cols = split_xy(df_feat, target, drop_cols)

    use_group = bool(group_col) and (group_col in df_feat.columns)
    if use_group:
        uniq = pd.Series(df_feat[group_col]).nunique()
        n_splits = int(min(10, max(2, uniq)))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(Xall, yall, groups=df_feat[group_col])
        cv_name = f"GroupKFold({n_splits})"
    else:
        if task == "cls":
            splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            splits = splitter.split(Xall, yall)
            cv_name = "StratifiedKFold(10, shuffle=True)"
        else:
            splitter = KFold(n_splits=10, shuffle=True, random_state=seed)
            splits = splitter.split(Xall, yall)
            cv_name = "KFold(10, shuffle=True)"

    rows = []
    oof_pred, oof_true, oof_idx, oof_fold, oof_proba = [], [], [], [], []
    fold_metrics_rows = []

    fold_idx = 0
    per_fold_dir = save_dir / "cv10"
    per_fold_dir.mkdir(parents=True, exist_ok=True)

    for tr_idx, va_idx in splits:
        fold_idx += 1
        Xtr, Xva = Xall.iloc[tr_idx], Xall.iloc[va_idx]
        ytr, yva = yall[tr_idx], yall[va_idx]

        base = build_rf(task=task, seed=seed+fold_idx, args=args)
        model = wrap_y_transform_if_needed(task, base, use_log1p=y_log1p)

        Xtr = Xtr.fillna(0.0); Xva = Xva.fillna(0.0)
        model.fit(Xtr, ytr)

        # ====== Save fold model and feature importance ======
        if save_fold_models:
            fold_model_path = per_fold_dir / f"fold_{fold_idx:02d}_model.joblib"
            payload_fold = {
                "model": model,
                "features": feat_cols,
                "target": target,
                "drop_cols": drop_cols,
                "y_log1p": bool(y_log1p) if task == "reg" else False,
                "group_col": group_col if (bool(group_col) and (group_col in df_feat.columns)) else None,
                "task": task,
                "decision_threshold": float(getattr(args, "decision_threshold", None)) if getattr(args, "decision_threshold", None) is not None else None
            }
            joblib.dump(payload_fold, fold_model_path)
            print("[saved]", fold_model_path)

            try:
                fi = getattr(model, "feature_importances_", None)
                if fi is None and hasattr(model, "regressor_"):
                    fi = getattr(model.regressor_, "feature_importances_", None)
                if fi is not None:
                    imp_fold = pd.DataFrame({"feature": feat_cols, "importance": fi})
                    imp_fold.sort_values("importance", ascending=False, inplace=True)
                    imp_fold.to_csv(per_fold_dir / f"feature_importance_fold_{fold_idx:02d}.csv",
                                    index=False, encoding="utf-8-sig")
                    print("[saved]", per_fold_dir / f"feature_importance_fold_{fold_idx:02d}.csv")
            except Exception as e:
                print(f"[warn] fold {fold_idx:02d} feature_importance export failed:", e)

        # ---- Validation Set ----
        if task == "reg":
            yhat_va = model.predict(Xva)
            m_va = eval_metrics_reg(yva, yhat_va)
            extra_va = {}
        else:
            proba_va = getattr(model, "predict_proba", lambda Z: None)(Xva)
            classes_ = getattr(model, "classes_", None)
            yhat_va_thr, yhat_va_argmax = predict_with_threshold_if_needed(
                model, Xva, getattr(args, "decision_threshold", None), classes_
            )
            yhat_va = yhat_va_thr if yhat_va_thr is not None else model.predict(Xva)

            m_va = eval_metrics_cls(yva, yhat_va, proba_va, classes_)
            m_va.update(per_class_accuracy(yva, yhat_va))
            extra_va = {}
            if proba_va is not None and classes_ is not None and len(classes_) == 2:
                pos = max(classes_)
                pos_idx = list(classes_).index(pos)
                extra_va["proba_pos"] = proba_va[:, pos_idx]
            if getattr(args, "decision_threshold", None) is not None and yhat_va_argmax is not None:
                extra_va["y_pred_argmax"] = yhat_va_argmax

        # ---- Training Set ----
        if task == "reg":
            yhat_tr = model.predict(Xtr)
            m_tr = eval_metrics_reg(ytr, yhat_tr)
            extra_tr = {}
        else:
            proba_tr = getattr(model, "predict_proba", lambda Z: None)(Xtr)
            classes_ = getattr(model, "classes_", None)
            yhat_tr_thr, yhat_tr_argmax = predict_with_threshold_if_needed(
                model, Xtr, getattr(args, "decision_threshold", None), classes_
            )
            yhat_tr = yhat_tr_thr if yhat_tr_thr is not None else model.predict(Xtr)

            m_tr = eval_metrics_cls(ytr, yhat_tr, proba_tr, classes_)
            m_tr.update(per_class_accuracy(ytr, yhat_tr))
            extra_tr = {}
            if proba_tr is not None and classes_ is not None and len(classes_) == 2:
                pos = max(classes_)
                pos_idx = list(classes_).index(pos)
                extra_tr["proba_pos"] = proba_tr[:, pos_idx]
            if getattr(args, "decision_threshold", None) is not None and yhat_tr_argmax is not None:
                extra_tr["y_pred_argmax"] = yhat_tr_argmax

        # Console print (keep old behavior: print validation metrics)
        metric_row = {"fold": fold_idx, **m_va}
        rows.append(metric_row)
        print(f"[cv10] fold {fold_idx}: {metric_row}")

        # OOF (Use Validation Set Labels)
        oof_pred.append(yhat_va)
        oof_true.append(yva)
        oof_idx.append(va_idx)
        oof_fold.append(np.full_like(va_idx, fill_value=fold_idx, dtype=int))
        if task == "cls":
            oof_proba.append(extra_va.get("proba_pos", None))

        # Per fold CSV
        df_tr_src = df_feat.iloc[tr_idx]
        df_va_src = df_feat.iloc[va_idx]

        dump_pred_table(
            df_src=df_tr_src,
            y_true=ytr,
            y_pred=yhat_tr,
            id_cols=id_cols,
            out_path=per_fold_dir / f"fold_{fold_idx:02d}_train.csv",
            extra_cols={"fold": np.full(len(tr_idx), fold_idx), **extra_tr}
        )

        dump_pred_table(
            df_src=df_va_src,
            y_true=yva,
            y_pred=yhat_va,
            id_cols=id_cols,
            out_path=per_fold_dir / f"fold_{fold_idx:02d}_valid.csv",
            extra_cols={"fold": np.full(len(va_idx), fold_idx), **extra_va}
        )

        with open(per_fold_dir / f"fold_{fold_idx:02d}_metrics.json", "w", encoding="utf-8") as fjs:
            json.dump({"fold": fold_idx, "train": m_tr, "valid": m_va}, fjs, ensure_ascii=False, indent=2)

        fold_metrics_rows.append({"fold": fold_idx, "split": "train", **m_tr})
        fold_metrics_rows.append({"fold": fold_idx, "split": "valid", **m_va})

    # Summary
    fold_metrics_csv = per_fold_dir / "cv10_fold_metrics.csv"
    pd.DataFrame(fold_metrics_rows).to_csv(fold_metrics_csv, index=False, encoding="utf-8-sig")
    print("[saved]", fold_metrics_csv)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(save_dir/"cv10_metrics.csv", index=False, encoding="utf-8-sig")
    print("[saved]", save_dir/"cv10_metrics.csv")

    if task == "reg":
        summary = {
            "cv": cv_name,
            "R2_mean": float(cv_df["R2"].mean()), "R2_std": float(cv_df["R2"].std(ddof=1) if len(cv_df)>1 else 0.0),
            "RMSE_mean": float(cv_df["RMSE"].mean()), "RMSE_std": float(cv_df["RMSE"].std(ddof=1) if len(cv_df)>1 else 0.0),
            "MAE_mean": float(cv_df["MAE"].mean()), "MAE_std": float(cv_df["MAE"].std(ddof=1) if len(cv_df)>1 else 0.0),
            "n_folds": int(len(cv_df))
        }
    else:
        def _m(col): return float(cv_df[col].mean()) if col in cv_df.columns else None
        def _s(col): return float(cv_df[col].std(ddof=1)) if col in cv_df.columns and len(cv_df)>1 else 0.0
        summary = {
            "cv": cv_name,
            "F1_macro_mean": _m("F1_macro"), "F1_macro_std": _s("F1_macro"),
            "BalAcc_mean": _m("BalAcc"), "BalAcc_std": _s("BalAcc"),
            "Acc_mean": _m("Acc"), "Acc_std": _s("Acc"),
            "ROC_AUC_mean": _m("ROC_AUC") if "ROC_AUC" in cv_df.columns else None,
            "ROC_AUC_ovr_mean": _m("ROC_AUC_ovr") if "ROC_AUC_ovr" in cv_df.columns else None,
            "n_folds": int(len(cv_df))
        }
        for c in np.unique(yall):
            col = f"Acc_class_{c}"
            if col in cv_df.columns:
                summary[f"{col}_mean"] = float(cv_df[col].mean())
                summary[f"{col}_std"]  = float(cv_df[col].std(ddof=1)) if len(cv_df) > 1 else 0.0

    with open(save_dir/"cv10_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[saved]", save_dir/"cv10_summary.json")
    print("[cv10] summary:", summary)

    # OOF Summary
    if save_train_pred:
        idx_all = np.concatenate(oof_idx)
        order = np.argsort(idx_all)
        y_true_all = np.concatenate(oof_true)[order]
        y_pred_all = np.concatenate(oof_pred)[order]
        fold_all   = np.concatenate(oof_fold)[order]
        df_src_all = df_feat.iloc[idx_all[order]]

        extra = {"fold": fold_all}
        if task == "cls" and any(o is not None for o in oof_proba):
            proba_list = []
            for pp, ord_idx in zip(oof_proba, oof_idx):
                if pp is None:
                    proba_list.append(np.full_like(ord_idx, np.nan, dtype=float))
                else:
                    proba_list.append(pp)
            proba_all = np.concatenate(proba_list)[order]
            extra["proba_pos"] = proba_all

        path = oof_path or str(save_dir / "cv10_oof.csv")
        dump_pred_table(
            df_src=df_src_all, y_true=y_true_all, y_pred=y_pred_all,
            id_cols=id_cols, out_path=path, extra_cols=extra
        )


# ============== Prediction on Test Set ==============

def predict_on_test(
    model_path: Path,
    test_feat_df: pd.DataFrame,
    id_cols: List[str],
    save_dir: Path,
    target_col: Optional[str] = None,
    pred_path: Optional[str] = None
):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feat_cols_ref: List[str] = bundle["features"]

    test_num = test_feat_df.select_dtypes(include=[np.number]).copy()
    for c in feat_cols_ref:
        if c not in test_num.columns:
            test_num[c] = 0.0
    Xt = test_num[feat_cols_ref].astype(float)

    y_true = None
    if target_col and (target_col in test_feat_df.columns):
        y_true = pd.to_numeric(test_feat_df[target_col], errors="coerce").to_numpy()

    # Prediction (supports threshold)
    y_pred = None
    y_pred_argmax = None

    decision_threshold = bundle.get("decision_threshold", None)
    classes_ = getattr(model, "classes_", None)

    if hasattr(model, "predict_proba") and decision_threshold is not None and classes_ is not None and len(classes_) == 2:
        P = model.predict_proba(Xt)
        pos = max(classes_)
        neg = min(classes_)
        pos_idx = list(classes_).index(pos)
        y_pred = np.where(P[:, pos_idx] >= float(decision_threshold), pos, neg)
        y_pred_argmax = model.predict(Xt)
    else:
        y_pred = model.predict(Xt)

    extra = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xt)
        classes_ = getattr(model, "classes_", None)
        if classes_ is not None and len(classes_) == 2:
            pos = max(classes_)
            pos_idx = list(classes_).index(pos)
            extra["proba_pos"] = proba[:, pos_idx]
    if y_pred_argmax is not None:
        extra["y_pred_argmax"] = y_pred_argmax

    out_path = pred_path or str(save_dir / "test_predictions.csv")
    dump_pred_table(df_src=test_feat_df, y_true=y_true, y_pred=y_pred,
                    id_cols=id_cols, out_path=out_path, extra_cols=extra)


# ============== Main Function and Arguments ==============

def main():
    ap = argparse.ArgumentParser(description="RandomForest Training Script (reg/cls, includes 10-fold CV and OOF export)")
    ap.add_argument("--in_csv", required=True, help="Feature CSV for training")
    ap.add_argument("--target", required=True, help="Target column name (continuous for regression; discretize labels for classification)")
    ap.add_argument("--task", choices=["reg","cls"], default="reg", help="Task type: regression (reg) or classification (cls)")
    ap.add_argument("--model", choices=["rf"], default="rf", help="Model type (rf only)")
    ap.add_argument("--save_dir", required=True, help="Output directory")
    ap.add_argument("--valid_size", type=float, default=0.2, help="Validation set ratio for single split (e.g., 0.25)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_cols", default="", help="Columns to drop during training (comma separated), e.g., 'Swelling Ratio (times),y_true'")
    ap.add_argument("--test_csv", default=None, help="Optional: Feature CSV for testing")
    ap.add_argument("--id_cols", default="", help="Columns to pass through in prediction output (comma separated)")

    # Target Transform (Regression only)
    ap.add_argument("--y_log1p", action="store_true", help="Regression: Train with log1p(target), restore with expm1 for prediction")

    # Grouping (Prevent Leakage)
    ap.add_argument("--group_col", default="", help="Grouping column name for GroupKFold (Optional)")
    ap.add_argument("--n_splits", type=int, default=5, help="Number of GroupKFold splits (Effective only for single split)")
    ap.add_argument("--fold_idx", type=int, default=0, help="Index of the fold to use as validation set (Effective only for single split)")

    # RF Hyperparameters (Robust defaults)
    ap.add_argument('--rf_n_estimators', type=int, default=900)
    ap.add_argument('--rf_max_depth', type=int, default=16)
    ap.add_argument('--rf_max_features', type=str, default=None,
                    help="Can be 0.3 (float), 128 (int), 'sqrt' or 'log2'; 'auto' maps to 1.0")
    ap.add_argument('--rf_min_samples_leaf', type=float, default=0.01,
                    help="Min samples per leaf, supports fraction (0,1] or int >= 1")
    ap.add_argument('--rf_min_samples_split', type=float, default=0.02,
                    help="Min samples required to split an internal node, int >= 2 or (0,1] fraction")
    ap.add_argument('--rf_class_weight', type=str, default="balanced_subsample",
                    help="Classification only, e.g., 'balanced'/'balanced_subsample'")
    ap.add_argument('--rf_max_samples', type=float, default=0.7,
                    help="(0,1] Subsample ratio for each tree (effective when bootstrap=True)")
    ap.add_argument('--rf_ccp_alpha', type=float, default=0.0, help="Complexity parameter for minimal cost-complexity pruning (>0 is more conservative)")
    ap.add_argument('--rf_min_impurity_decrease', type=float, default=0.0,
                    help="Threshold for early stopping, suggest trying 1e-4~1e-3")
    ap.add_argument('--rf_oob', action='store_true', help='Enable OOB estimation (oob_score=True)')

    # Low Variance Feature Dropping (Optional)
    ap.add_argument('--var_thresh', type=float, default=0.0, help='If >0, drop numeric columns with variance <= threshold')

    # 10-Fold CV & OOF Export
    ap.add_argument("--cv10", action="store_true", help="Enable 10-Fold Cross-Validation (Per-fold CSV/JSON & Summary CSV)")
    ap.add_argument("--save_train_pred", action="store_true", help="Save total training predictions (valid/train_predictions.csv; and cv10_oof.csv)")
    ap.add_argument("--pred_path", default=None, help="Custom path for test set prediction output (Default <save_dir>/test_predictions.csv)")
    ap.add_argument("--oof_path", default=None, help="Custom path for OOF prediction output (Default <save_dir>/cv10_oof.csv)")

    # New Features Switches
    ap.add_argument("--save_fold_models", action="store_true",
                    help="When using --cv10, save model for each fold to cv10/fold_XX_model.joblib and export feature importance")
    ap.add_argument("--decision_threshold", type=float, default=None,
                    help="Custom 'decision threshold' for binary classification (e.g., 0.35). If not set, use argmax of predict (equivalent to 0.5).")

    args = ap.parse_args()

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv_robust(args.in_csv)
    assert args.target in df.columns, f"Target column {args.target} not in training CSV"

    # Low variance column removal: only effective for numeric columns; aligned via drop_cols
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    if args.var_thresh and args.var_thresh > 0:
        num = df.select_dtypes(include=[np.number])
        variances = num.var(axis=0)
        low_cols = variances[variances <= args.var_thresh].index.tolist()
        low_cols = [c for c in low_cols if c != args.target]
        if low_cols:
            print(f"[var_thresh] drop {len(low_cols)} low-variance cols (≤ {args.var_thresh}):", low_cols[:10], "...")
            drop_cols += low_cols

    id_cols   = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    # ========= Single Split Training =========
    single_split_train_eval(
        task=args.task,
        df_feat=df,
        target=args.target,
        valid_size=args.valid_size,
        seed=int(args.seed),
        drop_cols=drop_cols,
        save_dir=save_dir,
        y_log1p=args.y_log1p,
        group_col=args.group_col,
        n_splits=int(args.n_splits),
        fold_idx=int(args.fold_idx),
        no_perm=False,
        save_train_pred=bool(args.save_train_pred),
        id_cols=id_cols,
        args=args,
    )

    # ========= Print CV Artifacts List (If exists) =========
    def _print_cv10_brief(save_dir: Path):
        p_sum   = save_dir / "cv10_summary.json"
        p_cv    = save_dir / "cv10_metrics.csv"
        p_fold  = save_dir / "cv10" / "cv10_fold_metrics.csv"
        p_oof   = save_dir / "cv10_oof.csv"
        paths   = [p_sum, p_cv, p_fold, p_oof]
        print("\n[cv10] artifacts:")
        for p in paths:
            print("  -", p, ("[OK]" if p.exists() else "[MISSING]"))
        if p_sum.exists():
            with open(p_sum, "r", encoding="utf-8") as f:
                summ = json.load(f)
            print("\n[cv10] summary:")
            keys_order = [
                "cv", "n_folds",
                "R2_mean","R2_std","RMSE_mean","RMSE_std","MAE_mean","MAE_std",
                "F1_macro_mean","F1_macro_std","BalAcc_mean","BalAcc_std",
                "Acc_mean","Acc_std","ROC_AUC_mean","ROC_AUC_ovr_mean",
            ]
            keys_order += [k for k in summ.keys() if k.startswith("Acc_class_") and k.endswith("_mean")]
            for k in keys_order:
                if k in summ and summ[k] is not None:
                    print(f"  {k}: {summ[k]}")
        print()

    # ========= 10-Fold Cross-Validation =========
    if args.cv10:
        cv10_evaluate_any(
            task=args.task,
            df_feat=df,
            target=args.target,
            seed=int(args.seed),
            drop_cols=drop_cols,
            save_dir=save_dir,
            y_log1p=args.y_log1p,
            group_col=args.group_col,
            save_train_pred=bool(args.save_train_pred),
            id_cols=id_cols,
            oof_path=args.oof_path,
            args=args,
            save_fold_models=bool(args.save_fold_models),
        )
        _print_cv10_brief(save_dir)

    # ========= Test Set Prediction (Optional) =========
    if args.test_csv:
        dft = read_csv_robust(args.test_csv)
        predict_on_test(
            model_path=save_dir/"best_model.joblib",
            test_feat_df=dft,
            id_cols=id_cols,
            save_dir=save_dir,
            target_col=args.target if args.target in dft.columns else None,
            pred_path=args.pred_path
        )


if __name__ == "__main__":
    main()