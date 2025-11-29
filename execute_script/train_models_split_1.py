#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import shap


RANDOM_STATE = 42

def load_features_and_labels(csv_path: str, label_col: str):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(...)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0.0)
    y_raw = df[label_col]
    return X, y_raw

def build_dt(mode: str):
    if mode == "binary":
        return DecisionTreeClassifier(
            criterion="entropy", max_leaf_nodes=315,
            class_weight="balanced", random_state=RANDOM_STATE
        )
    else:
        return DecisionTreeClassifier(
            criterion="gini", max_leaf_nodes=1001,
            class_weight="balanced", random_state=RANDOM_STATE
        )

def build_rf(mode: str, n_features: int):
    # Clip max_features to the available feature count
    if mode == "binary":
        mf = min(17, n_features)
        return RandomForestClassifier(
            n_estimators=10, max_leaf_nodes=851, max_features=mf,
            n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_STATE
        )
    else:
        mf = min(8, n_features)
        return RandomForestClassifier(
            n_estimators=54, max_leaf_nodes=1681, max_features=mf,
            n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_STATE
        )

def build_ann(mode: str):
    hidden = (79,) if mode == "binary" else (257,)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden, activation="logistic", solver="adam",
        max_iter=500, early_stopping=True, n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=True
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])

def weighted_fit(clf, X_train, y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {c: w for c, w in zip(classes, weights)}
    sample_weight = np.array([class_weight_dict[y_i] for y_i in y_train])
    try:
        clf.fit(X_train, y_train,
                **({"mlp__sample_weight": sample_weight} if isinstance(clf, Pipeline)
                   else {"sample_weight": sample_weight}))
    except TypeError:
        clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X, y, split_name: str):
    y_pred = clf.predict(X)
    return {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y, y_pred, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist()
    }
    
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def compute_tree_explanations(model, X_test_df, y_test_raw, le, out_prefix):
    feature_names = list(X_test_df.columns)

    # 1) Global: feature_importances_ (있으면)
    try:
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_)
            pd.DataFrame({
                "feature": feature_names,
                "importance": fi.astype(float)
            }).to_csv(f"{out_prefix}_feature_importances.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print("[WARN] feature_importances_ failed:", e)

    # 2) Global: permutation importance (옵션)
    try:


        perm = permutation_importance(
            model, X_test_df, le.transform(y_test_raw.astype(str)),
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
        )
        pd.DataFrame({
            "feature": feature_names,
            "mean":    perm.importances_mean.astype(float),
            "std":     perm.importances_std.astype(float)
        }).to_csv(f"{out_prefix}_permutation_importance.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print("[WARN] permutation_importance failed:", e)

    # 3) Per-sample SHAP (TreeExplainer) → CSV만 저장
    try:
        expl = shap.TreeExplainer(model)
        shap_vals_raw   = expl.shap_values(X_test_df)   # list (multiclass) 또는 ndarray (binary)
        base_values_raw = expl.expected_value

        # SHAP 반환 형태 정리
        if isinstance(shap_vals_raw, list):
            shap_list = [np.asarray(sv) for sv in shap_vals_raw]   # len = C_shap
        else:
            arr = np.asarray(shap_vals_raw)
            if arr.ndim == 2:                                      # (N,F) → binary
                shap_list = [arr]
            elif arr.ndim == 3:
                # 모양이 (C,N,F) 또는 (N,F,C)일 수 있어 양쪽 모두 시도
                if arr.shape[0] <= 20 and arr.shape[1] == len(X_test_df):
                    shap_list = [arr[c] for c in range(arr.shape[0])]
                elif arr.shape[-1] <= 20:
                    shap_list = [arr[:, :, c] for c in range(arr.shape[-1])]
                else:
                    raise RuntimeError(f"Unexpected SHAP 3D shape: {arr.shape}")
            else:
                raise RuntimeError(f"Unexpected SHAP shape: {arr.shape}")

        probs = model.predict_proba(X_test_df)                      # (N, C_probs)
        C_probs = probs.shape[1]
        C_shap  = len(shap_list)

        # 레퍼런스 클래스 누락 보정: C_shap == C_probs - 1 이면 마지막 클래스 복원
        if C_shap == C_probs - 1:
            missing = -np.sum(np.stack(shap_list, axis=0), axis=0)  # (N,F)
            shap_list.append(missing)
            C_shap = len(shap_list)

        # base value 정리
        if isinstance(base_values_raw, (list, np.ndarray)):
            base_arr = np.asarray(base_values_raw).reshape(-1)      # (C,) or (1,)
        else:
            base_arr = np.array([float(base_values_raw)])

        pred_labels_ix = np.argmax(probs, axis=1)
        pred_labels = [le.classes_[i] for i in pred_labels_ix]

        rows = []
        for i in range(len(X_test_df)):
            cls_ix = int(pred_labels_ix[i])
            # 만약 여전히 불일치가 있으면 clamp
            if cls_ix >= C_shap:
                cls_ix = C_shap - 1

            sv = np.asarray(shap_list[cls_ix][i]).ravel()           # (F,)
            base = float(base_arr[cls_ix] if base_arr.size > cls_ix else base_arr.flat[0])

            row = {
                "index": int(i),
                "true": str(y_test_raw.iloc[i]),
                "pred_label": str(pred_labels[i]),
                "pred_prob": float(probs[i, pred_labels_ix[i]]),
                "base_value": base
            }
            for j, fn in enumerate(feature_names):
                row[f"shap_{fn}"] = float(sv[j])
            rows.append(row)

        pd.DataFrame(rows).to_csv(f"{out_prefix}_shap.csv", index=False, encoding="utf-8-sig")
        print(f"[EXPLAIN][TREE] saved -> {out_prefix}_shap.csv")
    except Exception as e:
        print("compute_tree_explanations failed:", e)


def compute_ann_explanations(pipeline_model, X_train_df, X_test_df, y_test_raw, le, out_prefix, max_explain=200):
    """
    Save per-sample SHAP contributions for ANN (Pipeline[Scaler, MLP]) to CSV only.
    Output: f"{out_prefix}_shap.csv"
    """
    feature_names = list(X_test_df.columns)

    # Background as DataFrame (keeps feature names; avoids StandardScaler warnings)
    bg_size = min(100, len(X_train_df))
    background_df = X_train_df.sample(n=bg_size, random_state=RANDOM_STATE).reset_index(drop=True)

    # Prediction function that ALWAYS receives/returns with proper feature names
    def f(X):
        if isinstance(X, pd.DataFrame):
            X_df = X
            if list(X_df.columns) != feature_names:
                X_df = X_df[feature_names]
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            X_df = pd.DataFrame(X_arr, columns=feature_names)
        return pipeline_model.predict_proba(X_df)

    expl = shap.KernelExplainer(f, background_df, link="identity")

    # Limit #samples for speed
    n_to_explain = min(len(X_test_df), max_explain)
    subset_idx = list(range(n_to_explain))
    X_to_explain_df = X_test_df.iloc[subset_idx].reset_index(drop=True)

    shap_vals = expl.shap_values(X_to_explain_df, nsamples="auto")

    # Full-set predictions for meta info
    probs_full = pipeline_model.predict_proba(X_test_df)
    pred_labels_ix_full = np.argmax(probs_full, axis=1)
    pred_labels_full = [le.classes_[i] for i in pred_labels_ix_full]

    rows = []
    if isinstance(shap_vals, list):  # multiclass: list of (n_to_explain, n_features)
        for local_i, i in enumerate(subset_idx):
            cls_ix = int(pred_labels_ix_full[i])
            sv = np.asarray(shap_vals[cls_ix][local_i]).ravel()
            row = {
                "index": int(i),
                "true": str(y_test_raw.iloc[i]),
                "pred_label": str(pred_labels_full[i]),
                "pred_prob": float(probs_full[i, pred_labels_ix_full[i]]),
            }
            for j, fn in enumerate(feature_names):
                row[f"shap_{fn}"] = float(sv[j])
            rows.append(row)
    else:  # binary: array (n_to_explain, n_features)
        for local_i, i in enumerate(subset_idx):
            sv = np.asarray(shap_vals[local_i]).ravel()
            row = {
                "index": int(i),
                "true": str(y_test_raw.iloc[i]),
                "pred_label": str(pred_labels_full[i]),
                "pred_prob": float(probs_full[i, pred_labels_ix_full[i]]),
            }
            for j, fn in enumerate(feature_names):
                row[f"shap_{fn}"] = float(sv[j])
            rows.append(row)

    pd.DataFrame(rows).to_csv(f"{out_prefix}_shap.csv", index=False, encoding="utf-8-sig")
    print(f"[EXPLAIN][ANN] saved -> {out_prefix}_shap.csv  (rows={len(rows)}, features={len(feature_names)})")



# --- END ADDED helpers ---
def main():
    ap = argparse.ArgumentParser(description="Train DT, RF, ANN with pre-split CSV datasets")
    ap.add_argument("--mode", choices=["binary", "multi"], required=True, help="Choose task mode")
    ap.add_argument("--train_csv", required=True, help="Path to training CSV")
    ap.add_argument("--val_csv", required=True, help="Path to validation CSV")
    ap.add_argument("--test_csv", required=True, help="Path to test CSV")
    ap.add_argument("--label", default="NST_M_Label", help="Label column name")
    ap.add_argument("--outdir", default="outputs", help="Directory to save outputs")
    ap.add_argument("--explain", action="store_true", help="Save per-sample feature contributions on test set")
    ap.add_argument("--max_ann_explain", type=int, default=200, help="Max #test samples to explain for ANN (KernelSHAP)")
    args = ap.parse_args()

    outdir = Path(args.outdir); (outdir / "models").mkdir(parents=True, exist_ok=True)

    X_train, y_train_raw = load_features_and_labels(args.train_csv, args.label)
    X_val,   y_val_raw   = load_features_and_labels(args.val_csv,   args.label)
    X_test,  y_test_raw  = load_features_and_labels(args.test_csv,  args.label)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw.astype(str))
    y_val   = le.transform(y_val_raw.astype(str))
    y_test  = le.transform(y_test_raw.astype(str))

    # persist feature order & label mapping (for Pi inference)
    (outdir / "feature_order.json").write_text(
        json.dumps(list(X_train.columns), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    label_map = {int(i): str(c) for i, c in enumerate(le.classes_)}
    (outdir / "label_mapping.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dt = build_dt(args.mode)
    rf = build_rf(args.mode, n_features=X_train.shape[1])
    ann = build_ann(args.mode)

    results = {}
    expl_dir = outdir / "explanations"
    expl_dir.mkdir(parents=True, exist_ok=True)

    for key, model, name in [("DT", dt, "Decision Tree"),
                             ("RF", rf, "Random Forest"),
                             ("ANN", ann, "ANN (MLP)")]:
        model = weighted_fit(model, X_train, y_train)
        results[f"{key}_val"] = evaluate(model, X_val, y_val, "val")
        results[f"{key}_test"] = evaluate(model, X_test, y_test, "test")
        dump(model, outdir / "models" / f"{key.lower()}_{args.mode}.joblib")
        with open(outdir / f"{key.lower()}_{args.mode}_report.txt", "w", encoding="utf-8") as f:
            f.write(f"=== {name} ({args.mode}) ===\n")
            for split in ["val", "test"]:
                r = results[f"{key}_{split}"]
                f.write(f"[{split.upper()}] acc={r['accuracy']:.4f}, f1w={r['f1_weighted']:.4f}\n")
                f.write(r["report"] + "\n")
        # --- per-model explanations (옵션) ---
        if args.explain:
            model_path_prefix = str(expl_dir / f"{key.lower()}_{args.mode}")
            if key in ("DT", "RF"):
                # X_test는 DataFrame 형태 유지 필요
                compute_tree_explanations(model, X_test, y_test_raw, le, model_path_prefix)
            else:  # ANN (Pipeline)
                try:
                    compute_ann_explanations(#model.named_steps["mlp"].__class__ and model,
                                             model,
                                             X_train, X_test, y_test_raw, le, model_path_prefix,
                                             max_explain=args.max_ann_explain)
                except Exception as e:
                    print("ANN explanation failed:", e)
                    save_json({"error": str(e)}, f"{model_path_prefix}_shap_error.json")
    summary = {
        "mode": args.mode,
        "label_column": args.label,
        "classes": list(label_map.values()),
        "metrics": {k: {"val_acc": results[f"{k}_val"]["accuracy"],
                        "val_f1w": results[f"{k}_val"]["f1_weighted"],
                        "test_acc": results[f"{k}_test"]["accuracy"],
                        "test_f1w": results[f"{k}_test"]["f1_weighted"]}
                    for k in ["DT", "RF", "ANN"]}
    }
    with open(outdir / f"summary_{args.mode}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training done. Reports and models saved to", outdir)

if __name__ == "__main__":
    main()
