# 라벨 4개 포함안되는걸로 mrmr 계산
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ============================================
# 0) 출력 폴더 경로
# ============================================
OUTPUT_DIR = "/home/ice06/project/secure/mrmr_test/dataset/binary_icsflow/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# 1) 데이터 불러오기
# ============================================
DATA_PATH = "/home/ice06/project/secure/mrmr_test/dataset/Dataset.csv"
LABEL_COL = "NST_B_Label"  # 🔥 ICSFLOW:"NST_M_Label", FWA:" Label", IoT:"label", SWaT:"Normal/Attack"

df = (
    pd.read_csv(DATA_PATH)
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)


# -----------------------------
# 🔥 1-1) 모든 라벨 컬럼 자동 제거
# -----------------------------
# label 패턴: *_Label
label_candidates = [col for col in df.columns if "Label" in col]

print("Detected label columns:", label_candidates)

# y 선택
y_raw = df[LABEL_COL].astype(str)

# 🔥 feature set = 숫자형 + 라벨 제외 모든 feature
X = df.drop(columns=label_candidates, errors="ignore")
X = X.select_dtypes(include=[np.number])

print(f"Final feature count = {len(X.columns)}")


# Binary label: Benign = 0, Attack = 1
df["binary_label"] = df[LABEL_COL].apply(lambda x: 0 if x == "Benign" else 1)
y = df["binary_label"].values

""" 멀티 사용시
# Label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)
"""


feature_names = X.columns.tolist()
X_np = X.values

# ============================================
# 2) Relevance 계산 (Mutual Information)
# ============================================
print("Calculating Mutual Information (Relevance)...")

mi = mutual_info_classif(
    X_np,
    y,
    discrete_features=False
)

relevance_df = pd.DataFrame({
    "feature": feature_names,
    "mi": mi
}).sort_values("mi", ascending=False)

# 저장
relevance_path = os.path.join(OUTPUT_DIR, "relevance_sorted.csv")
relevance_df.to_csv(relevance_path, index=False, encoding="utf-8-sig")
print(f"Saved → {relevance_path}")

# ============================================
# 3) Redundancy 계산 함수
# ============================================
def mutual_info_pair(x1, x2, bins=30):
    """continuous MI between 2 features (safe for NaN/constant)"""
    x1 = np.asarray(x1).astype(float)
    x2 = np.asarray(x2).astype(float)

    if np.nanstd(x1) == 0 or np.nanstd(x2) == 0:
        return 0.0

    c1 = pd.cut(x1, bins=bins, labels=False)
    c2 = pd.cut(x2, bins=bins, labels=False)

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)

    mask = ~((np.isnan(c1)) | (np.isnan(c2)))
    c1 = c1[mask]
    c2 = c2[mask]

    if len(c1) == 0 or len(c2) == 0:
        return 0.0

    return mutual_info_score(c1, c2)

# ============================================
# 4) Redundancy 증가 곡선 계산
# ============================================
print("Calculating Redundancy Curve...")

sorted_features = relevance_df["feature"].tolist()
redundancy_curve = []
selected = []

for k in tqdm(range(1, len(sorted_features) + 1)):
    new_feat = sorted_features[k - 1]
    selected.append(new_feat)

    if len(selected) == 1:
        redundancy_curve.append(0.0)
    else:
        reds = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                reds.append(
                    mutual_info_pair(
                        df[selected[i]].values,
                        df[selected[j]].values
                    )
                )
        redundancy_curve.append(np.mean(reds))

redundancy_df = pd.DataFrame({
    "k": list(range(1, len(sorted_features) + 1)),
    "redundancy": redundancy_curve
})

# 저장
redundancy_path = os.path.join(OUTPUT_DIR, "redundancy_curve.csv")
redundancy_df.to_csv(redundancy_path, index=False, encoding="utf-8-sig")
print(f"Saved → {redundancy_path}")

# ============================================
# 5) Relevance 그래프
# ============================================
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(relevance_df) + 1), relevance_df["mi"].values, marker='o')
plt.title("Relevance (MI) vs Feature Rank")
plt.xlabel("Feature Rank (sorted by MI)")
plt.ylabel("Relevance (MI)")
plt.grid(True)
plt.tight_layout()

relevance_fig_path = os.path.join(OUTPUT_DIR, "relevance_curve.png")
plt.savefig(relevance_fig_path, dpi=200)
plt.show()
print(f"Saved → {relevance_fig_path}")

# ============================================
# 6) Redundancy 그래프
# ============================================
plt.figure(figsize=(12, 5))
plt.plot(redundancy_df["k"], redundancy_df["redundancy"], marker='o', color='red')
plt.title("Redundancy vs k")
plt.xlabel("k (#selected features)")
plt.ylabel("Avg redundancy (MI between features)")
plt.grid(True)
plt.tight_layout()

redundancy_fig_path = os.path.join(OUTPUT_DIR, "redundancy_curve.png")
plt.savefig(redundancy_fig_path, dpi=200)
plt.show()
print(f"Saved → {redundancy_fig_path}")

print("\nAll results saved to:", OUTPUT_DIR)