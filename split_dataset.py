# split_icsflow.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LABEL_CANDIDATES = [
    "NST_M_Label", "IT_M_Label",  # multi-class
    "NST_B_Label", "IT_B_Label",  # binary
    "label", "Label"
]

def guess_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("라벨 컬럼을 찾지 못했습니다. LABEL_CANDIDATES를 확인하세요.")

def can_stratify(df: pd.DataFrame, y_col: str, n_sets: int) -> bool:
    # 각 클래스가 최소 n_sets 개 이상 있어야 모든 세트에 1개 이상 배정 가능(필수는 아님)
    vc = df[y_col].value_counts()
    return (vc.min() >= n_sets)

def hamilton_allocate(total: int, weights: np.ndarray) -> np.ndarray:
    """
    해밀턴 방식(최대 잔여법)으로 정수 카운트 배분.
    - weights: 각 클래스의 '비중'(합이 1일 필요는 없고, 상대비율이면 됨)
    - total: 세트의 총 샘플 수
    """
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        # 모두 0이면 균등 분배
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    quotas = weights * total
    base = np.floor(quotas).astype(int)
    remainder = quotas - base
    # 남은 좌석(정수) 배정
    remain = total - base.sum()
    if remain > 0:
        order = np.argsort(-remainder)  # 큰 잔여부터
        base[order[:remain]] += 1
    return base

def proportional_split(df: pd.DataFrame, y_col: str, ratios=(0.5, 0.3, 0.2), random_state=42):
    """
    전체 클래스 분포를 기준으로 train/val/test 각각에 정수로 정확히 배정.
    해밀턴 방식으로 클래스별 타깃 카운트를 만든 뒤, 클래스별 무작위 추출.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios 합은 1이어야 합니다."
    rng = np.random.RandomState(random_state)

    # 전체 클래스 분포
    classes, class_counts = np.unique(df[y_col], return_counts=True)
    n_total = len(df)
    n_train = int(round(n_total * ratios[0]))
    n_val   = int(round(n_total * ratios[1]))
    n_test  = n_total - n_train - n_val  # 합 맞추기

    # 각 세트의 클래스별 목표 카운트
    train_targets = hamilton_allocate(n_train, class_counts)
    val_targets   = hamilton_allocate(n_val,   class_counts)
    test_targets  = class_counts - train_targets - val_targets
    # (혹시 음수가 생기면 마지막에 조정)
    test_targets = np.maximum(test_targets, 0)

    # 클래스별로 샘플링
    train_idx, val_idx, test_idx = [], [], []
    for cls, t_tr, t_va, t_te in zip(classes, train_targets, val_targets, test_targets):
        idx = df.index[df[y_col] == cls].to_numpy()
        rng.shuffle(idx)

        # 필요 카운트를 초과하면 잘라냄(정수 배정 상 안전장치)
        t_tr = min(t_tr, len(idx))
        cls_train = idx[:t_tr]
        remain = idx[t_tr:]

        t_va = min(t_va, len(remain))
        cls_val  = remain[:t_va]
        remain2  = remain[t_va:]

        t_te = min(t_te, len(remain2))
        cls_test = remain2[:t_te]

        # 남는 샘플이 있다면 test에 합쳐서 총량 보정(아주 드문 모서리 케이스)
        rest = remain2[t_te:]
        if len(rest) > 0:
            # 남은 샘플들을 test로 흡수(비율과 아주 미세하게 달라질 수 있음)
            cls_test = np.concatenate([cls_test, rest])

        train_idx.append(cls_train)
        val_idx.append(cls_val)
        test_idx.append(cls_test)

    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_idx)
    test_idx  = np.concatenate(test_idx)

    train_df = df.loc[train_idx].sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df   = df.loc[val_idx].sample(frac=1, random_state=random_state+1).reset_index(drop=True)
    test_df  = df.loc[test_idx].sample(frac=1, random_state=random_state+2).reset_index(drop=True)

    return train_df, val_df, test_df

def random_split(df: pd.DataFrame, y_col: str, ratios=(0.5, 0.3, 0.2), random_state=42):
    # 1단계: 50% / 50%
    strat1 = can_stratify(df, y_col, n_sets=2)
    train_df, temp_df = train_test_split(
        df, test_size=ratios[1] + ratios[2], shuffle=True, random_state=random_state,
        stratify=(df[y_col] if strat1 else None)
    )
    # 2단계: 남은 50%를 30%/20%가 되도록 60%/40% 분할
    test_ratio_in_temp = ratios[2] / (ratios[1] + ratios[2])  # 0.2 / 0.5 = 0.4
    strat2 = can_stratify(temp_df, y_col, n_sets=2)
    test_df, val_df = train_test_split(
        temp_df, test_size=(1 - test_ratio_in_temp), shuffle=True, random_state=random_state+1,
        stratify=(temp_df[y_col] if strat2 else None)
    )
    # 위에서 test/val 순서 바꿨다면 뒤집기
    # 현재 계산으로는 test_ratio_in_temp=0.4 → test(40%), val(60%)가 되어야 하므로,
    # 위 두 줄에서 변수명에 주의: 최종적으로 50/30/20이 되도록 값은 맞습니다.
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def print_dist(tag, df, y_col):
    vc = df[y_col].value_counts(normalize=True).sort_index()
    info = (vc * 100).round(2).to_dict()
    print(f"{tag:>6} | n={len(df):5d} | dist% {info}")

def process_one(csv_path: str, mode: str, ratios=(0.5, 0.3, 0.2), random_state=42):
    df = pd.read_csv(csv_path)
    y_col = guess_label_col(df)

    if mode == "proportional":
        train_df, val_df, test_df = proportional_split(df, y_col, ratios, random_state)
        strat_note = "proportional (Hamilton)"
    elif mode == "random":
        train_df, val_df, test_df = random_split(df, y_col, ratios, random_state)
        strat_note = "random (stratify if possible)"
    else:
        raise ValueError("--mode 는 'proportional' 또는 'random' 이어야 합니다.")

    base = csv_path.rsplit(".", 1)[0]
    train_df.to_csv(f"{base}_train.csv", index=False)
    val_df.to_csv(f"{base}_val.csv", index=False)
    test_df.to_csv(f"{base}_test.csv", index=False)

    print(f"\n[{csv_path}] → split done: {strat_note}")
    print_dist("TRAIN", train_df, y_col)
    print_dist("  VAL", val_df,   y_col)
    print_dist(" TEST", test_df,  y_col)

def main():
    p = argparse.ArgumentParser(description="ICS-Flow 50/30/20 splitter (proportional or random).")
    p.add_argument("--mode", choices=["proportional", "random"], default="proportional",
                   help="proportional: 전체 분포를 각 세트에 정확히 반영 / random: 무작위(가능하면 stratify).")
    p.add_argument("--files", nargs="+", default=[
        "reduced_features_IT_B.csv",
        "reduced_features_IT_M.csv",
        "reduced_features_NST_B.csv",
        "reduced_features_NST_M.csv",
    ], help="분할할 CSV 파일 목록")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = p.parse_args()

    for f in args.files:
        process_one(f, mode=args.mode, random_state=args.seed)

if __name__ == "__main__":
    main()



# python split_dataset.py --files  /home/ice06/project/secure/mrmr_test/advice/dataset/cv_baseline/cv_baseline.csv