import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm

# ============================================
# 0) 설정값
# ============================================

DATASET = "/home/ice06/project/secure/hyewon/advice/dataset/Modbus_dataset/selected_ics_45719.csv"
OUTPUT_DIR = "/home/ice06/project/secure/mrmr_test/dataset/new_modbus/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 라벨 컬럼 (feature selection에서 절대 포함되면 안 됨)
LABEL_COLS = [
    "NST_M_Label", "NST_B_Label",
    "IT_M_Label", "IT_B_Label"
]

# MISS 파라미터
N_BOOTSTRAP = 30 # 몇 번 섞을지
SAMPLE_FRAC = 0.7 # 매번 몇 개 뽑을지
TOP_K_EACH_ROUND = 30
PROB_THRESHOLD = 0.7
RANDOM_STATE = 42

# ============================================
# 1) 데이터 불러오기
# ============================================

print("📂 Loading dataset...")
df = pd.read_csv(DATASET).fillna(0)

print(f"- Total samples in dataset: {len(df)}")

# ============================================
# 2) 라벨 컬럼 완전 제거 (강화 버전)
# ============================================

# 대소문자 무시를 위한 lowercase 리스트
LABEL_COLS_LOWER = [c.lower() for c in LABEL_COLS]

# df 내부에서 라벨로 간주되는 컬럼 자동 탐지
detected_label_cols = [c for c in df.columns if c.lower() in LABEL_COLS_LOWER]

print("🚫 Excluding label columns from feature candidates:")
print(detected_label_cols)

# feature candidates = 라벨 컬럼 제거 + 숫자형만 남기기
feature_df = df.drop(columns=detected_label_cols, errors='ignore')
feature_df = feature_df.select_dtypes(include=[np.number])

# 다시 한 번 라벨이 들어있는지 안전검사
bad_cols = [c for c in feature_df.columns if c.lower() in LABEL_COLS_LOWER]
if len(bad_cols) > 0:
    raise ValueError(f"❌ ERROR: label columns detected in feature set → {bad_cols}")

feature_names = feature_df.columns.tolist()
X_np = feature_df.values

print(f"- Feature candidates after removal: {len(feature_names)}")

# ============================================
# 3) 라벨 인코딩 (NST_M_Label 기준)
# ============================================

label_for_miss = "NST_M_Label"

if label_for_miss not in df.columns:
    raise ValueError("❌ NST_M_Label is missing in the dataset!")

y_raw = df[label_for_miss].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ============================================
# 4) MISS 알고리즘 정의
# ============================================

def miss_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
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

    print("🚀 Running MISS (Mutual Information + Stability Selection)...")
    for b in tqdm(range(n_bootstrap)):
        idx = rng.choice(n_samples, size=int(n_samples * sample_frac), replace=True)
        X_b = X[idx]
        y_b = y[idx]

        mi = mutual_info_classif(
            X_b, y_b, 
            discrete_features=False,
            random_state=rng.randint(0, 999999)
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
        by=["selection_prob", "avg_mi"], ascending=[False, False]
    ).reset_index(drop=True)

    selected_df = result_df[result_df["selection_prob"] >= prob_threshold]

    if len(selected_df) == 0:
        print("⚠️ No feature meets threshold → fallback to top 20")
        selected_df = result_df.head(20)

    return selected_df["feature"].tolist(), result_df, selected_df

# ============================================
# 5) MISS 실행
# ============================================

selected_features, result_df, selected_df = miss_feature_selection(
    X_np, y, feature_names,
    n_bootstrap=N_BOOTSTRAP,
    sample_frac=SAMPLE_FRAC,
    top_k_each_round=TOP_K_EACH_ROUND,
    prob_threshold=PROB_THRESHOLD,
    random_state=RANDOM_STATE
)

print("\n✅ MISS selected features:")
for f in selected_features:
    print(" -", f)

print(f"\n총 선택된 피처 수: {len(selected_features)}")

# ============================================
# 6) 결과 저장
# ============================================

result_df.to_csv(os.path.join(OUTPUT_DIR, "miss_feature_scores.csv"), index=False)
selected_df.to_csv(os.path.join(OUTPUT_DIR, "miss_selected_features.csv"), index=False)

# 최종 데이터셋 구성: 선택된 Feature + 4개 Label 컬럼 유지
df_reduced = df[selected_features + LABEL_COLS]
df_reduced.to_csv(os.path.join(OUTPUT_DIR, "Dataset_miss_selected.csv"), index=False)

print("💾 Saved Dataset_miss_selected.csv")
