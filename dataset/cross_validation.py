import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

##############################
# 경로 설정
##############################
DATASET = "Dataset.csv"
RELEVANCE_FILE = "/home/ice06/project/secure/mrmr_test/advice/dataset/icsflow/mrmr3/relevance_sorted.csv"
OUTPUT_DIR = "/home/ice06/project/secure/mrmr_test/advice/dataset/mrmr_method3/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

##############################
# 데이터 불러오기
##############################
df = pd.read_csv(DATASET).fillna(0)
label_col = "NST_M_Label"   # 🔥 multi-class로 변경
y = LabelEncoder().fit_transform(df[label_col].astype(str))

# 중요도 순서대로 feature 불러오기
rel_df = pd.read_csv(RELEVANCE_FILE)
sorted_features = rel_df["feature"].tolist()

##############################
# 모델 정의
##############################
def build_models():
    dt = DecisionTreeClassifier(max_leaf_nodes=500, class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1)
    ann = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))
    ])
    return dt, rf, ann

##############################
# 평가 루프 (k=1 ~ 전체)
##############################
results = []

for k in tqdm(range(1, len(sorted_features) + 1)): # 전체 보고 싶으면 len(sorted_features) + 1
    feats = sorted_features[:k]
    X = df[feats]

    # train/val split (8:2)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt, rf, ann = build_models()

    # DT
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_val)

    # RF
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_val)

    # ANN
    ann.fit(X_train, y_train)
    pred_ann = ann.predict(X_val)

    results.append({
        "k": k,
        "features": ",".join(feats),   # ⭐ 추가: 이 k에서 사용한 feature 목록
        "DT_acc": accuracy_score(y_val, pred_dt),
        "DT_f1": f1_score(y_val, pred_dt, average="weighted"),
        "RF_acc": accuracy_score(y_val, pred_rf),
        "RF_f1": f1_score(y_val, pred_rf, average="weighted"),
        "ANN_acc": accuracy_score(y_val, pred_ann),
        "ANN_f1": f1_score(y_val, pred_ann, average="weighted"),
    })

##############################
# 결과 저장
##############################
res_df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "k_performance.csv")
res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("Saved →", csv_path)

##############################
# 그래프 그리기
##############################
plt.figure(figsize=(12,6))
plt.plot(res_df["k"], res_df["RF_f1"], label="RF F1", marker='o')
plt.plot(res_df["k"], res_df["DT_f1"], label="DT F1", marker='o')
plt.plot(res_df["k"], res_df["ANN_f1"], label="ANN F1", marker='o')
plt.xlabel("k (top-k features)")
plt.ylabel("F1 score (weighted)")
plt.title("Cross-validation performance vs k")
plt.grid(True)
plt.legend()
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "performance_vs_k.png")
plt.savefig(fig_path, dpi=200)
plt.show()

print("Saved →", fig_path)

##############################
# 최적 k 찾기
##############################
best_row = res_df.loc[res_df["RF_f1"].idxmax()]   # RF 기준
best_k = int(best_row["k"])
best_feats = sorted_features[:best_k]

print("\n🔥 최적 k =", best_k)
print(best_row)

print("\n✅ 최적 k에서 사용된 feature 목록:")
for f in best_feats:
    print("  -", f)


"""
🔥 최적 k = 22
k                                                          22
features    end,rBytesSum,endOffset,start,sBytesSum,startO...
DT_acc                                               0.992235
DT_f1                                                0.992221
RF_acc                                               0.995407
RF_f1                                                0.995388
ANN_acc                                              0.982612
ANN_f1                                               0.981778
Name: 21, dtype: object

✅ 최적 k에서 사용된 feature 목록:
  - end
  - rBytesSum
  - endOffset
  - start
  - sBytesSum
  - startOffset
  - rPayloadSum
  - sPayloadSum
  - rPackets
  - sPackets
  - rBytesAvg
  - rInterPacketAvg
  - sLoad
  - rLoad
  - sInterPacketAvg
  - rPayloadAvg
  - rAckDelayAvg
  - rPshRate
  - sAckDelayAvg
  - sBytesAvg
  - rAckDelayMax
  - sPayloadAvg

"""