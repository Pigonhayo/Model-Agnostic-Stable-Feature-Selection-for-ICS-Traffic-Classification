#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42


def load_features_and_labels(csv_path: str, label_col: str, feature_order=None):
    """
    테스트 CSV에서 피처(X)와 라벨(y_raw)을 로드.
    feature_order가 주어지면, 학습 때와 동일한 컬럼 순서로 맞춰줌.
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    # 숫자 피처만 사용
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0.0)

    # feature_order가 있으면 그 순서로 정렬 (없는 컬럼은 0으로 채움)
    if feature_order is not None:
        X = X.reindex(columns=feature_order).fillna(0.0)

    y_raw = df[label_col]
    return X, y_raw


def evaluate(clf, X, y, split_name: str):
    """
    간단 평가 함수: acc, f1_weighted, report, confusion matrix.
    """
    y_pred = clf.predict(X)
    return {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y, y_pred, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Zero-shot TEST only: load pre-trained DT/RF/ANN and evaluate on test CSV."
    )
    ap.add_argument("--mode", choices=["binary", "multi"], required=True,
                    help="Task mode (binary or multi). 모델 파일명에 반영됨.")
    ap.add_argument("--test_csv",required=True, help="Path to test CSV")
    ap.add_argument("--label", default="NST_B_label",  # ← CSV 안에서 라벨 컬럼 이름
                help="Label column name in CSV")
    ap.add_argument("--outdir", default="outputs",
                help="Directory that already contains models/ + feature_order.json + label_mapping.json")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    models_dir = outdir / "models"

    print("[MODE] ZERO-SHOT TEST ONLY")
    print(f" - mode      : {args.mode}")
    print(f" - test_csv  : {args.test_csv}")
    print(f" - outdir    : {outdir}")

    # 1) feature_order, label_mapping 로드 (학습 시 생성된 파일)
    feat_path = outdir / "feature_order.json"
    label_map_path = outdir / "label_mapping.json"

    if not feat_path.exists() or not label_map_path.exists():
        raise FileNotFoundError(
            f"feature_order.json 또는 label_mapping.json 을 찾을 수 없습니다.\n"
            f"먼저 학습 스크립트를 실행해서 {outdir} 아래에 해당 파일들을 생성해야 합니다."
        )

    feature_order = json.loads(feat_path.read_text(encoding="utf-8"))
    label_map_raw = json.loads(label_map_path.read_text(encoding="utf-8"))

    # idx -> label(str)
    idx_to_label = {int(k): v for k, v in label_map_raw.items()}
    classes_sorted = [idx_to_label[i] for i in sorted(idx_to_label.keys())]

    le = LabelEncoder()
    le.classes_ = np.array(classes_sorted, dtype=object)

    # 2) 테스트 데이터 로드 (피처 순서 맞추기)
    X_test, y_test_raw = load_features_and_labels(
        args.test_csv, args.label, feature_order=feature_order
    )
    y_test = le.transform(y_test_raw.astype(str))

    print(f"[INFO] Loaded test set: n={len(X_test)}, n_features={X_test.shape[1]}")
    print(f"[INFO] Classes: {list(le.classes_)}")

    # 3) 학습된 모델 로드
    dt_path = models_dir / f"dt_{args.mode}.joblib"
    rf_path = models_dir / f"rf_{args.mode}.joblib"
    ann_path = models_dir / f"ann_{args.mode}.joblib"

    if not dt_path.exists() or not rf_path.exists() or not ann_path.exists():
        raise FileNotFoundError(
            f"필요한 모델 파일들을 찾을 수 없습니다.\n"
            f"기대 위치:\n"
            f"  - {dt_path}\n"
            f"  - {rf_path}\n"
            f"  - {ann_path}\n"
            f"먼저 학습 스크립트를 실행해서 해당 joblib 파일들을 생성해야 합니다."
        )

    print("[INFO] Loading models...")
    dt = load(dt_path)
    rf = load(rf_path)
    ann = load(ann_path)

    # 4) 평가 & 결과 저장
    results = {}
    for key, model, name in [
        ("DT", dt, "Decision Tree"),
        ("RF", rf, "Random Forest"),
        ("ANN", ann, "ANN (MLP)"),
    ]:
        print(f"\n[TEST] Evaluating {name} ...")
        r = evaluate(model, X_test, y_test, "test")
        results[f"{key}_test"] = r

        # 간단 출력
        print(f"  - acc={r['accuracy']:.4f}, f1_weighted={r['f1_weighted']:.4f}")
        # 리포트 파일 저장
        report_path = outdir / f"{key.lower()}_{args.mode}_zeroshot_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== {name} ({args.mode}) [ZERO-SHOT TEST] ===\n")
            f.write(f"[TEST] acc={r['accuracy']:.4f}, f1w={r['f1_weighted']:.4f}\n\n")
            f.write(r["report"] + "\n")
        print(f"  → saved report to {report_path}")

    # 5) 요약 JSON 저장
    summary = {
        "mode": args.mode,
        "label_column": args.label,
        "classes": classes_sorted,
        "metrics": {
            k: {
                "test_acc": results[f"{k}_test"]["accuracy"],
                "test_f1w": results[f"{k}_test"]["f1_weighted"],
                "confusion_matrix": results[f"{k}_test"]["confusion_matrix"],
            }
            for k in ["DT", "RF", "ANN"]
        },
    }
    summary_path = outdir / f"summary_{args.mode}_zeroshot.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Zero-shot TEST summary saved to {summary_path}")


if __name__ == "__main__":
    main()

"""
python train_models_split.py \
  --mode binary \
  --test_csv /home/ice06/project/secure/hyewon/advice/dataset/Modbus_dataset/reduce_14_test.csv \
  --label NST_B_Label \
  --outdir /home/ice06/project/secure/hyewon/advice/outputs/NST_B/feat14
"""