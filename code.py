import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

import warnings
warnings.filterwarnings("ignore")

# =====================
# Load data
# =====================
train = pd.read_csv("/kaggle/input/train.csv")
test = pd.read_csv("/kaggle/input/test.csv")

TEXT_COL = "essay"
TARGET = "Importance Score"

X_text = train[TEXT_COL].astype(str)
y = train[TARGET].values
X_test_text = test[TEXT_COL].astype(str)

# =====================
# TF-IDF
# =====================
tfidf = TfidfVectorizer(
    max_features=200000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =====================
# SVD
# =====================
svd = TruncatedSVD(
    n_components=300,
    random_state=42
)

X_svd = svd.fit_transform(X_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

# =====================
# Scale SVD
# =====================
scaler = StandardScaler()
X_svd = scaler.fit_transform(X_svd)
X_test_svd = scaler.transform(X_test_svd)

# =====================
# Combine features
# =====================
X_final = hstack([X_tfidf, X_svd])
X_test_final = hstack([X_test_tfidf, X_test_svd])

# =====================
# K-Fold
# =====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

# =====================
# Models
# =====================
ridge = Ridge(alpha=1.0)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    random_state=42,
    n_jobs=-1
)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)

# =====================
# Training loop
# =====================
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_final)):
    print(f"\nFold {fold + 1}")

    X_tr, X_val = X_final[tr_idx], X_final[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # Ridge
    ridge.fit(X_tr, y_tr)
    ridge_val = ridge.predict(X_val)
    ridge_test = ridge.predict(X_test_final)

    # LightGBM
    lgb_model.fit(X_tr, y_tr)
    lgb_val = lgb_model.predict(X_val)
    lgb_test = lgb_model.predict(X_test_final)

    # XGBoost
    xgb_model.fit(X_tr, y_tr)
    xgb_val = xgb_model.predict(X_val)
    xgb_test = xgb_model.predict(X_test_final)

    # Ensemble
    val_pred = (ridge_val + lgb_val + xgb_val) / 3
    test_pred = (ridge_test + lgb_test + xgb_test) / 3

    oof_preds[val_idx] = val_pred
    test_preds += test_pred / kf.n_splits

    rmse = mean_squared_error(y_val, val_pred, squared=False)
    print("RMSE:", rmse)

# =====================
# Final RMSE
# =====================
final_rmse = mean_squared_error(y, oof_preds, squared=False)
print("\nFinal CV RMSE:", final_rmse)

# =====================
# Submission
# =====================
submission = pd.DataFrame({
    "id": test["id"],
    "Importance Score": test_preds
})

submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv saved")

