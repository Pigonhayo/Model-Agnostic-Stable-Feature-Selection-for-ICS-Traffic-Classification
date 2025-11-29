############################################################
#  MA-SRF: Model-Agnostic Stability & Redundancy Feature Selection
#  Standalone version (mRMR 전처리 방식 + ANN 안정화 수정버전)
############################################################

import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold


############################################################
# clone with random_state
############################################################
def clone_with_seed(estimator, seed):
    est = clone(estimator)
    def set_seed(obj):
        if hasattr(obj, "random_state"):
            obj.random_state = seed
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                set_seed(step)
    set_seed(est)
    return est


############################################################
# MA-SRF Main Algorithm
############################################################
def ma_srf_feature_selection(
    X, y,
    n_bootstrap=10,
    top_percent=0.3,
    cv_splits=5,
    eps=0.002,
    random_state=42,
    verbose=True
):

    feature_names = X.columns.tolist()
    X_np = X.values
    n_samples, n_features = X_np.shape
    rng = np.random.RandomState(random_state)

    ########################################################
    # 1) Base Models (DT, RF, ANN)
    ########################################################
    models = {
        "DT": DecisionTreeClassifier(random_state=random_state),

        "RF": RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=random_state
        ),

        # ANN 안정화 버전
        "ANN": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64,),   # 가벼운 모델
                activation="relu",
                solver="adam",
                max_iter=120,
                random_state=random_state
            ))
        ])
    }

    weights = {"DT": 0.4, "RF": 0.4, "ANN": 0.2}

    mean_imp = {m: np.zeros(n_features) for m in models}
    stability = {m: np.zeros(n_features) for m in models}
    n_top = max(1, int(np.ceil(top_percent * n_features)))

    ########################################################
    # 2) Bootstrapped Importance + Stability
    ########################################################
    for m_name, model in models.items():
        if verbose:
            print(f"[MA-SRF] Bootstrapping importance → {m_name}")

        # ANN은 부트스트랩 횟수 줄임
        if m_name == "ANN":
            actual_bootstrap = 3     # ANN 10 → 3
        else:
            actual_bootstrap = n_bootstrap

        all_imps = []

        for b in range(actual_bootstrap):
            idx = rng.choice(n_samples, int(0.8 * n_samples), replace=True)
            X_b, y_b = X_np[idx], y[idx]

            mdl = clone_with_seed(model, random_state + b)
            mdl.fit(X_b, y_b)

            # ANN은 PI 반복과 병렬 줄임
            if m_name == "ANN":
                rep = 2              # 5 → 2
                jobs = 1             # n_jobs=-1 → 1
            else:
                rep = 5
                jobs = -1

            imp = permutation_importance(
                mdl, X_b, y_b,
                scoring="f1_weighted",
                n_repeats=rep,
                random_state=random_state + b,
                n_jobs=jobs
            ).importances_mean

            all_imps.append(imp)

            top_idx = np.argsort(imp)[::-1][:n_top]
            stability[m_name][top_idx] += 1

        all_imps = np.vstack(all_imps)
        mean_imp[m_name] = all_imps.mean(axis=0)
        stability[m_name] /= actual_bootstrap

    ########################################################
    # 3) Ensemble Importance
    ########################################################
    ensemble = np.zeros(n_features)
    for m in models:
        ensemble += weights[m] * (mean_imp[m] * stability[m])
    ensemble = np.maximum(ensemble, 0)

    ########################################################
    # 4) Correlation-based Redundancy Penalty
    ########################################################
    Xc = X_np - X_np.mean(axis=0)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(Xc, rowvar=False)
    corr = np.nan_to_num(corr)

    remaining = set(range(n_features))
    selected = []
    final_score = np.zeros(n_features)

    print("[MA-SRF] Applying redundancy penalty ...")

    while remaining:
        best_idx = None
        best_val = -np.inf

        for j in remaining:
            penalty = 1.0 if not selected else (1 - max(abs(corr[j, k]) for k in selected))
            val = ensemble[j] * penalty

            if val > best_val:
                best_val = val
                best_idx = j

        selected.append(best_idx)
        remaining.remove(best_idx)
        final_score[best_idx] = best_val

    feature_order = [feature_names[i] for i in selected]

    ########################################################
    # 5) Optimal k by CV Performance
    ########################################################
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_mean = -999
    score_table = []

    for k in range(1, n_features + 1):
        idx_sel = selected[:k]
        X_sel = X_np[:, idx_sel]

        scores = []
        for model in models.values():
            mdl = clone_with_seed(model, random_state + 999)
            sc = cross_val_score(
                mdl, X_sel, y,
                scoring="f1_weighted",
                cv=skf,
                n_jobs=-1
            ).mean()
            scores.append(sc)

        mean_f1 = np.mean(scores)
        score_table.append([k, mean_f1])

        if mean_f1 > best_mean:
            best_mean = mean_f1

    valid_k = [k for k, f1 in score_table if f1 >= best_mean - eps]
    selected_k = min(valid_k)

    selected_features = feature_order[:selected_k]

    return {
        "selected_features": selected_features,
        "selected_k": selected_k,
        "feature_order": feature_order,
        "final_importance": final_score[selected],
        "score_table": pd.DataFrame(score_table, columns=["k", "mean_f1"])
    }


############################################################
# MAIN
############################################################
if __name__ == "__main__":

    ########################################################
    # 여기를 네 경로로 수정하면 끝
    ########################################################
    DATA_PATH = "/home/ice06/project/secure/mrmr_test/advice/dataset/Dataset.csv"
    LABEL_COL = "NST_M_Label"
    OUTDIR = "/home/ice06/project/secure/mrmr_test/advice/dataset/ma_srf"
    ########################################################

    print("\n=========================================")
    print("     MA-SRF Feature Selection Start")
    print("=========================================\n")

    df = pd.read_csv(DATA_PATH).fillna(0)

    # 모든 Label 컬럼 자동 탐지
    label_cols = [col for col in df.columns if "Label" in col]
    print("Detected label columns:", label_cols)

    if LABEL_COL not in df.columns:
        raise KeyError(f"지정한 LABEL_COL({LABEL_COL})이 df에 존재하지 않습니다!")

    # y 선택
    y_raw = df[LABEL_COL].astype(str)
    y = LabelEncoder().fit_transform(y_raw)

    # X = label 전체 제거 + 숫자형만
    X = df.drop(columns=label_cols, errors="ignore")
    X = X.select_dtypes(include=["number"])

    print("Final feature count =", X.shape[1])

    os.makedirs(OUTDIR, exist_ok=True)

    result = ma_srf_feature_selection(X, y, verbose=True)

    # 저장
    pd.Series(result["selected_features"]).to_csv(
        os.path.join(OUTDIR, "MA_SRF_selected_features.txt"),
        index=False
    )
    pd.Series(result["feature_order"]).to_csv(
        os.path.join(OUTDIR, "MA_SRF_feature_order.txt"),
        index=False
    )
    result["score_table"].to_csv(
        os.path.join(OUTDIR, "MA_SRF_scores_by_k.csv"),
        index=False
    )

    print("\n====================== DONE ======================")
    print("Selected k:", result["selected_k"])
    print("Results saved →", OUTDIR)
    print("===================================================")
