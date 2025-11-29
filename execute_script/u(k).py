# ============================================
# u_k.py  (Utility-based k selection)
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# -------------------------------
# 1) U(k) 함수 정의
# -------------------------------
def compute_Uk(Pk_list, k, p, lam=0.1):
    """
    U(k) = P(k) - λ * (k/p)
    """
    return Pk_list[k-1] - lam * (k / p)


# -------------------------------
# 2) k별 성능 계산 함수
# -------------------------------
def evaluate_Pk(df, sorted_features, label_col, model=None, n_splits=5):
    """
    df: 전체 DataFrame
    sorted_features: mRMR 등으로 정렬된 feature 리스트
    label_col: 라벨 컬럼명
    model: 사용할 ML 모델 (기본 RF)
    """

    X_all = df.drop(columns=[label_col])
    y = df[label_col].astype(str)
    y = y.astype("category").cat.codes  # encoding

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    p = len(sorted_features)
    Pk_list = []

    print(f"\n[INFO] Calculating P(k) for k=1..{p}")

    for k in tqdm(range(1, p + 1)):
        selected = sorted_features[:k]
        X = X_all[selected].values

        # CV 평가
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            f1_scores.append(f1_score(y_val, preds, average="macro"))

        Pk_list.append(np.mean(f1_scores))

    return Pk_list


# -------------------------------
# 3) 최적 k 찾기
# -------------------------------
def find_best_k(Pk_list, lam=0.1):
    p = len(Pk_list)
    Uk_list = []

    for k in range(1, p + 1):
        Uk_list.append(Pk_list[k-1] - lam * (k / p))

    best_k = np.argmax(Uk_list) + 1
    return best_k, Uk_list


# -------------------------------
# 4) 실행 메인 함수
# -------------------------------
def run_u_k(
    data_path,
    ranking_path,
    label_col="NST_M_Label",
    lam=0.1,
    n_splits=5
):

    print("\n==============================")
    print(" Running U(k) Feature Selection ")
    print("==============================")

    # 데이터 로드
    df = pd.read_csv(data_path)
    ranking_df = pd.read_csv(ranking_path)

    sorted_features = ranking_df["feature"].tolist()

    # 1) k별 성능(P(k)) 계산
    Pk_list = evaluate_Pk(df, sorted_features, label_col, n_splits=n_splits)

    # 2) U(k) 계산 → 최적 k 찾기
    best_k, Uk_list = find_best_k(Pk_list, lam=lam)

    print("\n[RESULT] P(k) =", Pk_list)
    print("[RESULT] U(k) =", Uk_list)
    print(f"\n🔥 Best k = {best_k}")

    return best_k, Pk_list, Uk_list


# -------------------------------
# 5) 직접 실행할 때
# -------------------------------
if __name__ == "__main__":
    # ====== 사용자 설정 ======
    DATA_PATH = "/home/ice06/project/secure/mrmr_test/advice/Dataset.csv"
    RANKING_PATH = "/home/ice06/project/secure/mrmr_test/advice/dataset/icsflow/mrmr/relevance_sorted.csv"
    LABEL = "NST_M_Label"

    best_k, Pk_list, Uk_list = run_u_k(
        data_path=DATA_PATH,
        ranking_path=RANKING_PATH,
        label_col=LABEL,
        lam=0.1,
        n_splits=5
    )
