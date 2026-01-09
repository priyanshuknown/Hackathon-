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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading Data...")
train = pd.read_csv("/kaggle/input/bash-8-0-round-2/train.csv").fillna('')
test = pd.read_csv("/kaggle/input/bash-8-0-round-2/test.csv").fillna('')

print(f"Train Shape: {train.shape}")
print(f"Test Shape:  {test.shape}")

# 2. Feature Engineering & Preprocessing
print("Engineering Features...")

# Create unified text columns
train['text_blob'] = train['Headline'] + " " + train['Key Insights'] + " " + train['Reasoning'] + " " + train['Power Mentions']
test['text_blob'] = test['Headline'] + " " + test['Key Insights'] + " " + test['Reasoning'] + " " + test['Power Mentions']

train['headline_insights'] = train['Headline'] + " " + train['Key Insights']
test['headline_insights'] = test['Headline'] + " " + test['Key Insights']

train['reasoning_full'] = train['Reasoning'] + " " + train['Tags']
test['reasoning_full'] = test['Reasoning'] + " " + test['Tags']

# Custom Counts - semicolon separated
count_cols = ['Power Mentions', 'Agencies', 'Lead Types', 'Tags']
for c in count_cols:
    train[f'{c}_cnt'] = train[c].apply(lambda x: x.count(';') + 1 if x.strip() else 0)
    test[f'{c}_cnt'] = test[c].apply(lambda x: x.count(';') + 1 if x.strip() else 0)

# Flag VIPs
vips = ['Epstein', 'Clinton', 'Trump', 'Maxwell', 'Prince', 'Dershowitz', 'Acosta', 'Barak', 'Wexner']
for name in vips:
    train[f'has_{name}'] = train['text_blob'].str.contains(name, case=False).astype(int)
    test[f'has_{name}'] = test['text_blob'].str.contains(name, case=False).astype(int)

# Basic Text stats
train['len'] = train['text_blob'].apply(len)
train['word_len'] = train['text_blob'].apply(lambda x: len(x.split()))
test['len'] = test['text_blob'].apply(len)
test['word_len'] = test['text_blob'].apply(lambda x: len(x.split()))

# 3. Vectorization (TF-IDF + SVD)
print("Vectorizing Text...")

# Main TF-IDF
vec = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3), 
                      sublinear_tf=True, min_df=2, max_df=0.95)
X_tfidf_tr = vec.fit_transform(train['text_blob'])
X_tfidf_te = vec.transform(test['text_blob'])

# Secondary TF-IDF
vec2 = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
X_tfidf_tr2 = vec2.fit_transform(train['headline_insights'])
X_tfidf_te2 = vec2.transform(test['headline_insights'])

# SVD reduction
svd = TruncatedSVD(n_components=150, random_state=42)
X_svd_tr = svd.fit_transform(X_tfidf_tr)
X_svd_te = svd.transform(X_tfidf_te)

svd2 = TruncatedSVD(n_components=75, random_state=42)
X_svd_tr2 = svd2.fit_transform(X_tfidf_tr2)
X_svd_te2 = svd2.transform(X_tfidf_te2)

# 4. Prepare Feature Matrices
print("Preparing Matrices...")

# Identify numeric features
num_feats = [c for c in train.columns if '_cnt' in c or 'has_' in c or '_len' in c or c in ['len', 'word_len', 'capital_ratio', 'exclamation_cnt', 'question_cnt']]

# Scaling
scaler = StandardScaler()
num_tr_sc = scaler.fit_transform(train[num_feats])
num_te_sc = scaler.transform(test[num_feats])

# Dense features (Numerics + SVD) - For Trees
X_tr_dense = np.hstack([train[num_feats].values, X_svd_tr, X_svd_tr2])
X_te_dense = np.hstack([test[num_feats].values, X_svd_te, X_svd_te2])

# Sparse features (TFIDF + Scaled Numerics) - For Linear Models
X_tr_sparse = hstack([X_tfidf_tr, X_tfidf_tr2, num_tr_sc])
X_te_sparse = hstack([X_tfidf_te, X_tfidf_te2, num_te_sc])

y = train['Importance Score'].values

# 5. Modeling (KFold Ensemble)
print("Starting Training...")

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof = np.zeros(len(train))
preds = np.zeros(len(test))
scores = []

for fold, (idx_tr, idx_val) in enumerate(kf.split(X_tr_dense)):
    # Split Data
    X_tr_d, X_val_d = X_tr_dense[idx_tr], X_tr_dense[idx_val]
    X_tr_s, X_val_s = X_tr_sparse.tocsr()[idx_tr], X_tr_sparse.tocsr()[idx_val]
    y_tr, y_val = y[idx_tr], y[idx_val]

    # Model 1: LightGBM
    clf1 = lgb.LGBMRegressor(
        n_estimators=7000, learning_rate=0.1, num_leaves=64, 
        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    clf1.fit(X_tr_d, y_tr, eval_set=[(X_val_d, y_val)], eval_metric='rmse', 
             callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    val1 = clf1.predict(X_val_d)
    test1 = clf1.predict(X_te_dense)

    # Model 2: XGBoost
    clf2 = xgb.XGBRegressor(
        n_estimators=3500, learning_rate=0.1, max_depth=8,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=0.1, tree_method='hist',
        random_state=92, n_jobs=-1, early_stopping_rounds=100
    )
    clf2.fit(X_tr_d, y_tr, eval_set=[(X_val_d, y_val)], verbose=False)
    val2 = clf2.predict(X_val_d)
    test2 = clf2.predict(X_te_dense)
    
    # Model 3: Ridge
    clf3 = Ridge(alpha=1.0)
    clf3.fit(X_tr_s, y_tr)
    val3 = clf3.predict(X_val_s)
    test3 = clf3.predict(X_te_sparse)

    # Ensemble Weights (50% LGBM, 40% XGB, 10% Ridge)
    ens_val = (0.50 * val1) + (0.40 * val2) + (0.10 * val3)
    oof[idx_val] = ens_val
    
    # Accumulate test predictions (Average over folds)
    preds += ((0.50 * test1) + (0.40 * test2) + (0.10 * test3)) / N_SPLITS
    
    rmse = mean_squared_error(y_val, ens_val)**0.5
    scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

print(f"\nMean RMSE: {np.mean(scores):.4f}")

# 6. Submission
submission = pd.DataFrame({'id': test['id'], 'Importance Score': np.clip(preds, 0, 100)})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved 'submission.csv'")
