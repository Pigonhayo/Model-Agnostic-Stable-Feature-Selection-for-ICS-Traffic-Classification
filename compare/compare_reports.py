#!/usr/bin/env python3
import re
import pandas as pd
from pathlib import Path

# === 경로 설정 ===
BASE_DIR = "/home/ice06/project/secure/hyewon/advice/outputs/NST_M/feature23"       # 논문
CUR_DIR = "/home/ice06/project/secure/hyewon/advice/outputs/NST_M/final7"            # 각자 파일에 맞게 수정
OUT_FILE = "/home/ice06/project/secure/hyewon/advice/outputs/NST_M/final7/compare_final7.csv" # 각자 파일에 맞게 수정

MODELS = ["ann", "dt", "rf"]  # 비교할 모델 리스트


def parse_report(file_path, section="TEST"):
    """
    classification_report.txt에서 [TEST] 섹션만 파싱
    """
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    # TEST 섹션 시작 위치 찾기
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"[{section}]"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"{section} 섹션을 찾을 수 없음: {file_path}")

    # precision, recall, f1-score 부분만 추출
    rows = []
    for line in lines[start_idx+1:]:
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 4 and parts[0] not in ("precision", "accuracy", "macro", "weighted", ""):
            cls, prec, rec, f1 = parts[:4]
            try:
                rows.append({
                    "class": cls,
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1)
                })
            except ValueError:
                continue
        if line.strip().startswith("accuracy"):
            break  # TEST 블록 끝나면 중단
    return pd.DataFrame(rows)


def compare_models(base_dir, cur_dir, models):
    """
    base_dir 과 cur_dir 의 각 모델별 report를 비교 (TEST 부분만)
    """
    results = None
    for model in models:
        base_file = Path(base_dir) / f"{model}_multi_report.txt"
        cur_file = Path(cur_dir) / f"{model}_multi_report.txt"

        base_df = parse_report(base_file, section="TEST")
        cur_df = parse_report(cur_file, section="TEST")

        diff_df = cur_df.copy()
        diff_df["precision"] = cur_df["precision"] - base_df["precision"]
        diff_df["recall"] = cur_df["recall"] - base_df["recall"]
        diff_df["f1"] = cur_df["f1"] - base_df["f1"]

        # 컬럼 이름 변경 (모델 이름 prefix)
        diff_df = diff_df.rename(columns={
            "precision": f"{model.upper()}_precision",
            "recall": f"{model.upper()}_recall",
            "f1": f"{model.upper()}_f1"
        })

        if results is None:
            results = diff_df
        else:
            results = pd.merge(results, diff_df, on="class", how="outer")

    return results


if __name__ == "__main__":
    df = compare_models(BASE_DIR, CUR_DIR, MODELS)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[DONE] 비교 결과 저장됨 → {OUT_FILE}")
