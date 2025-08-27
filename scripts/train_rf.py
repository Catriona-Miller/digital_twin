import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


df_raw = pd.read_csv('../data/combined_matrix_large.tsv', sep='\t')

meta = pd.read_csv('../data/meta.tsv', sep='\t')
df_raw = df_raw[df_raw['subjectID'].isin(meta[meta['Condition'] == 'MAM']['subjectID'])]

targets = meta.set_index('subjectID')['Recovery']
df_raw = df_raw.set_index('subjectID')
targets = targets.apply(lambda x: 0 if x == "No recovery" else 1)

# only keep targets that subject_id is in df_raw
targets = targets[targets.index.isin(df_raw.index)]

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_raw, targets, test_size=0.2, random_state=42, stratify=targets
)

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', class_weight_dict]
}

# Initialize RandomizedSearchCV
rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit and get best model
rf_random.fit(X_train, y_train)
rf_model = rf_random.best_estimator_

print("Best parameters:", rf_random.best_params_)
print("Best CV ROC-AUC score:", rf_random.best_score_)

# Evaluate the model
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, rf_model.predict(X_test))
print(f"Test AUC Score: {auc_score:.4f}")
print(f"Test F1 Score: {f1:.4f}")
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, rf_model.predict(X_test))
print(f"AUC Score: {auc_score:.4f}")
print(f"F1 Score: {f1:.4f}")