import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# ============================================
# 0) 설정값
# ============================================

DATASET = "/home/ice06/project/secure/mrmr_test/dataset/Dataset.csv"
OUTPUT_DIR = "/home/ice06/project/secure/mrmr_test/dataset/new"

# 라벨로 사용되는 컬럼들(기여도에서 제외)
LABEL_COLS = [
    "NST_M_Label", "NST_B_Label",
    "IT_M_Label",  "IT_B_Label"
]

# feature selection에 사용할 라벨 값 (예: multi-class의 일부만 사용)
ALLOWED_LABELS = ["Normal", "ddos", "ip-scan", "port-scan"]  # 너가 원하는 라벨 subset

os.makedirs(OUTPUT_DIR, exist_ok=True)

# MISS 파라미터
N_BOOTSTRAP = 30
SAMPLE_FRAC = 0.7
TOP_K_EACH_ROUND = 30
PROB_THRESHOLD = 0.7
RANDOM_STATE = 42

# ============================================
# 1) 원본 데이터 로드
# ============================================
print("📂 Loading dataset...")
df_full = pd.read_csv(DATASET).fillna(0)

# ============================================
# 2) feature selection용 서브셋 만들기
#    → 딱 하나의 라벨 컬럼(NST_M_Label) 기준으로 필터링
# ============================================
filter_label = "NST_M_Label"

df = df_full[df_full[filter_label].isin(ALLOWED_LABELS)].reset_index(drop=True)

print(f"🎯 Feature selection using rows where {filter_label} in {ALLOWED_LABELS}")
print(f"- Original samples: {len(df_full)}")
print(f"- Filtered samples: {len(df)}")

# ============================================
# 3) feature 컬럼만 추출
# ============================================

# 라벨 컬럼들을 제외한 나머지 숫자형 feature만 사용
feature_df = df.drop(columns=LABEL_COLS, errors='ignore').select_dtypes(include=[np.number])

feature_names = feature_df.columns.tolist()
X = feature_df.values

# 라벨 인코딩 (MISS에서 사용)
y_raw = df[filter_label].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"- Total candidate features: {len(feature_names)}")

# ============================================
# 4) MISS 알고리즘 정의
# ============================================

def miss_feature_selection(
    X, y, feature_names,
    n_bootstrap=30,
    sample_frac=0.7,
    top_k_each_round=30,
    prob_threshold=0.7,
    random_state=42,
):
    n_samples, n_features = X.shape
    select_counts = np.zeros(n_features, dtype=int)
    mi_sums = np.zeros(n_features)

    rng = np.random.RandomState(random_state)

    print("🚀 Running MISS (Stability + Mutual Information)")
    for b in tqdm(range(n_bootstrap)):
        idx = rng.choice(n_samples, size=int(n_samples * sample_frac), replace=True)
        X_b = X[idx]
        y_b = y[idx]

        mi = mutual_info_classif(
            X_b, y_b, discrete_features=False,
            random_state=rng.randint(0, 99999)
        )

        top_k = min(top_k_each_round, n_features)
        top_idx = np.argsort(mi)[::-1][:top_k]

        select_counts[top_idx] += 1
        mi_sums += mi

    selection_prob = select_counts / n_bootstrap
    avg_mi = mi_sums / n_bootstrap

    result_df = pd.DataFrame({
        "feature": feature_names,
        "select_count": select_counts,
        "selection_prob": selection_prob,
        "avg_mi": avg_mi
    }).sort_values(
        by=["selection_prob", "avg_mi"],
        ascending=[False, False]
    ).reset_index(drop=True)

    selected_df = result_df[result_df["selection_prob"] >= prob_threshold]

    if len(selected_df) == 0:
        print("⚠️ No features passed threshold → fallback to top 20")
        selected_df = result_df.head(20)

    return selected_df["feature"].tolist(), result_df, selected_df

# ============================================
# 5) MISS 실행
# ============================================
selected_features, result_df, selected_df = miss_feature_selection(
    X, y, feature_names,
    n_bootstrap=N_BOOTSTRAP,
    sample_frac=SAMPLE_FRAC,
    top_k_each_round=TOP_K_EACH_ROUND,
    prob_threshold=PROB_THRESHOLD,
    random_state=RANDOM_STATE
)

print("\n✅ Selected Features:")
for f in selected_features:
    print("-", f)

# ============================================
# 6) 결과 저장
# ============================================
result_df.to_csv(os.path.join(OUTPUT_DIR, "miss_feature_scores.csv"), index=False)
selected_df.to_csv(os.path.join(OUTPUT_DIR, "miss_selected_features.csv"), index=False)

# 전체 데이터셋에서 선택된 피처만 남기기 (라벨은 전부 유지)
df_reduced = df_full[selected_features + LABEL_COLS]
df_reduced.to_csv(os.path.join(OUTPUT_DIR, "Dataset_miss_selected.csv"), index=False)

print("💾 Saved Dataset_miss_selected.csv")
