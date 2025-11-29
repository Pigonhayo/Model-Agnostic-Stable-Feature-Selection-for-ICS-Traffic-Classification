#!/usr/bin/env bash
set -euo pipefail
#source "./execute_script/hyperparams.sh"
#source "./execute_script/config_paths.sh"

#!/usr/bin/env bash
# =====================
# Paths & labels
# =====================
MODE="multi"                        # "binary" | "multi"
TRAIN_CSV="/home/ice06/project/secure/mrmr_test/dataset/check_new/check_new_train.csv"  # 경로 각자 맞춰서 쓰세요.
VAL_CSV="/home/ice06/project/secure/mrmr_test/dataset/check_new/check_new_val.csv"      # 경로 각자 맞춰서 쓰세요.
TEST_CSV="/home/ice06/project/secure/mrmr_test/dataset/check_new/check_new_test.csv"    # 경로 각자 맞춰서 쓰세요.
LABEL_COL="NST_M_Label"             #NST_M_Label or "B" for binary
OUTDIR="outputs/NST_M/check_new"        # 경로 각자 맞춰서 쓰세요.

PYTHON="python3"                    # python binary selector
TRAIN_PY="./execute_script/train_models_split_1.py"    # training script path
PI_PREDICT_PY="./execute_script/pi_predict.py"       # inference script path (for Raspberry Pi)

#!/usr/bin/env bash
# =====================
# Hyperparameters (paper-mapped) for reference & logging
# =====================

# Decision Tree
DT_BINARY_CRITERION="entropy"    # (paper: Maximum deviance reduction)
DT_BINARY_MAX_SPLITS=314         # -> sklearn max_leaf_nodes=315
DT_MULTI_CRITERION="gini"        # (paper: Twoing -> approximated as gini)
DT_MULTI_MAX_SPLITS=1000        # -> sklearn max_leaf_nodes=1001

# Random Forest
RF_BINARY_N_ESTIMATORS=10
RF_BINARY_MAX_SPLITS=850         # -> max_leaf_nodes=851
RF_BINARY_MAX_FEATURES=17        # clipped to n_features if smaller
RF_MULTI_N_ESTIMATORS=54
RF_MULTI_MAX_SPLITS=1680         # -> max_leaf_nodes=1681
RF_MULTI_MAX_FEATURES=8          # clipped to n_features if smaller

# ANN (MLP)
ANN_ACTIVATION="sigmoid"        # sigmoid
ANN_BINARY_HIDDEN=79
ANN_MULTI_HIDDEN=257
ANN_SOLVER="adam"
ANN_MAX_ITER=500

echo "=== Training with pre-split CSVs ==="
echo "MODE=$MODE  LABEL_COL=$LABEL_COL"
echo "TRAIN=$TRAIN_CSV"
echo "VAL  =$VAL_CSV"
echo "TEST =$TEST_CSV"
echo "OUTDIR=$OUTDIR"

# Log hyperparameters for traceability (actual values are implemented in Python)
cat <<LOG
[DT binary]   criterion=$DT_BINARY_CRITERION, max_splits=$DT_BINARY_MAX_SPLITS
[DT multi]    criterion=$DT_MULTI_CRITERION,  max_splits=$DT_MULTI_MAX_SPLITS
[RF binary]   n_estimators=$RF_BINARY_N_ESTIMATORS, max_splits=$RF_BINARY_MAX_SPLITS, max_features=$RF_BINARY_MAX_FEATURES
[RF multi]    n_estimators=$RF_MULTI_N_ESTIMATORS, max_splits=$RF_MULTI_MAX_SPLITS, max_features=$RF_MULTI_MAX_FEATURES
[ANN]         activation=$ANN_ACTIVATION, binary_hidden=$ANN_BINARY_HIDDEN, multi_hidden=$ANN_MULTI_HIDDEN, solver=$ANN_SOLVER, max_iter=$ANN_MAX_ITER
LOG

$PYTHON "$TRAIN_PY" \
  --mode "$MODE" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --label "$LABEL_COL" \
  --outdir "$OUTDIR" \
  --explain
