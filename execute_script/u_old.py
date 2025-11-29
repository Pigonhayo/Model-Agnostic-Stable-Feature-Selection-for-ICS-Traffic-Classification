import pandas as pd
import numpy as np
import os

# ===== 경로 설정 =====
OUTPUT_DIR = "/home/ice06/project/secure/mrmr_test/advice/dataset/icsflow/mrmr"

relevance_path = os.path.join(OUTPUT_DIR, "relevance_sorted.csv")
redundancy_path = os.path.join(OUTPUT_DIR, "redundancy_curve.csv")

# 1) Relevance (각 피처 MI, 중요도순 정렬)
rel_df = pd.read_csv(relevance_path)  # columns: feature, mi
# 이미 MI 내림차순 정렬된 상태라고 가정

# 2) Redundancy (k별 평균 redundancy)
red_df = pd.read_csv(redundancy_path)  # columns: k, redundancy

# 3) k별 평균 Relevance 계산 (상위 k개 피처의 평균 MI)
rel_df["mean_relevance_k"] = rel_df["mi"].expanding().mean()
# k=1 → 첫 피처 MI
# k=2 → 상위 2개 피처 MI의 평균
# ...

# k 인덱스 붙이기
rel_df["k"] = np.arange(1, len(rel_df) + 1)

# 4) 두 데이터프레임 merge
merged = pd.merge(rel_df[["k", "mean_relevance_k"]],
                  red_df[["k", "redundancy"]],
                  on="k",
                  how="inner")

# 5) 원래 스타일 U(k) = Relevance(k) - Redundancy(k)
merged["U_old"] = merged["mean_relevance_k"] - merged["redundancy"]

# 6) 최적 k 찾기
best_row = merged.loc[merged["U_old"].idxmax()]
best_k = int(best_row["k"])
best_U = best_row["U_old"]

print(merged.head())
print("\n[RESULT] Best k (original mRMR-style U(k)) =", best_k)
print("[RESULT] U_old(k) at best k =", best_U)

# 원하면 csv로 저장
out_path = os.path.join(OUTPUT_DIR, "mrmr_Uk_curve.csv")
merged.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved → {out_path}")
