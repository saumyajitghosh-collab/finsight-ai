"""
Banking AI Platform - ML Model Training Pipeline
Trains and saves models for:
  1. Credit Risk Scoring (Gradient Boosting)
  2. Fraud Detection (XGBoost)
  3. KYC/AML Risk Classifier (Random Forest)
  4. Churn Prediction (Gradient Boosting)
Also computes model metrics and saves them for the dashboard.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

metrics = {}
encoders = {}

print("=" * 60)
print("BANKING AI PLATFORM - MODEL TRAINING")
print("=" * 60)

# ============================================================
# 1. CREDIT RISK SCORING MODEL (Gradient Boosting)
# ============================================================
print("\n[1/4] Training Credit Risk Scoring Model...")
loans = pd.read_csv(DATA_DIR / "loans.csv")

credit_features = ['loan_amount', 'interest_rate', 'tenure_months', 'age',
                   'annual_income', 'months_on_book', 'num_products',
                   'loan_to_income', 'emi_to_income', 'income_log',
                   'loan_amount_log', 'credit_bureau_score',
                   'employment_type', 'loan_type', 'segment']

X_credit = loans[credit_features].copy()
y_credit = loans['default']

# Encode categorical features
for col in ['employment_type', 'loan_type', 'segment']:
    le = LabelEncoder()
    X_credit[col] = le.fit_transform(X_credit[col].astype(str))
    encoders[f'credit_{col}'] = le

X_train, X_test, y_train, y_test = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

credit_model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='auc', use_label_encoder=False)
credit_model.fit(X_train_bal, y_train_bal)

y_pred = credit_model.predict(X_test)
y_proba = credit_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

joblib.dump(credit_model, MODEL_DIR / "credit_risk_model.pkl")
joblib.dump(encoders, MODEL_DIR / "credit_encoders.pkl")
joblib.dump(credit_features, MODEL_DIR / "credit_features.pkl")

metrics['credit_risk'] = {
    'auc': round(auc, 4),
    'accuracy': round(report['accuracy'], 4),
    'precision': round(report['1']['precision'], 4),
    'recall': round(report['1']['recall'], 4),
    'f1': round(report['1']['f1-score'], 4),
    'confusion_matrix': cm.tolist(),
    'feature_count': len(credit_features),
    'train_samples': len(X_train_bal),
    'test_samples': len(X_test),
    'default_rate': float(y_credit.mean()),
}
# Feature importance
fi = pd.DataFrame({'feature': credit_features,
                   'importance': credit_model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
metrics['credit_risk']['feature_importance'] = fi.to_dict('records')

print(f"   ✓ AUC: {auc:.4f} | Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f}")

# ============================================================
# 2. FRAUD DETECTION MODEL (XGBoost)
# ============================================================
print("\n[2/4] Training Fraud Detection Model...")
txns = pd.read_csv(DATA_DIR / "transactions.csv")

fraud_features = ['amount', 'amount_log', 'txn_type', 'channel', 'merchant',
                  'hour_of_day', 'day_of_week', 'is_international',
                  'is_night_txn', 'is_weekend', 'is_high_amount']

X_fraud = txns[fraud_features].copy()
y_fraud = txns['is_fraud']

for col in ['txn_type', 'channel', 'merchant']:
    le = LabelEncoder()
    X_fraud[col] = le.fit_transform(X_fraud[col].astype(str))
    encoders[f'fraud_{col}'] = le

X_train, X_test, y_train, y_test = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

# SMOTE for fraud class
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

fraud_model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric='auc', use_label_encoder=False)
fraud_model.fit(X_train_bal, y_train_bal)

y_pred = fraud_model.predict(X_test)
y_proba = fraud_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

joblib.dump(fraud_model, MODEL_DIR / "fraud_model.pkl")
joblib.dump(encoders, MODEL_DIR / "fraud_encoders.pkl")  # overwrite with all encoders
# Save all encoders together
joblib.dump(encoders, MODEL_DIR / "all_encoders.pkl")
joblib.dump(fraud_features, MODEL_DIR / "fraud_features.pkl")

metrics['fraud_detection'] = {
    'auc': round(auc, 4),
    'accuracy': round(report['accuracy'], 4),
    'precision': round(report['1']['precision'], 4),
    'recall': round(report['1']['recall'], 4),
    'f1': round(report['1']['f1-score'], 4),
    'confusion_matrix': cm.tolist(),
    'feature_count': len(fraud_features),
    'train_samples': len(X_train_bal),
    'test_samples': len(X_test),
    'fraud_rate': float(y_fraud.mean()),
}
fi = pd.DataFrame({'feature': fraud_features,
                   'importance': fraud_model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
metrics['fraud_detection']['feature_importance'] = fi.to_dict('records')

print(f"   ✓ AUC: {auc:.4f} | Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f}")

# ============================================================
# 3. KYC/AML RISK CLASSIFIER (Random Forest)
# ============================================================
print("\n[3/4] Training KYC/AML Risk Classification Model...")
kyc = pd.read_csv(DATA_DIR / "kyc_aml.csv")

kyc_features = ['entity_type', 'country', 'doc_type', 'doc_verified',
                'num_bank_accounts', 'txn_volume_monthly', 'has_pep_link',
                'has_sanction_link', 'adverse_media_hits', 'years_in_business',
                'ownership_transparency', 'txn_volume_log', 'high_risk_country']

X_kyc = kyc[kyc_features].copy()
y_kyc = kyc['alert_flag']

for col in ['entity_type', 'country', 'doc_type', 'ownership_transparency']:
    le = LabelEncoder()
    X_kyc[col] = le.fit_transform(X_kyc[col].astype(str))
    encoders[f'kyc_{col}'] = le

joblib.dump(encoders, MODEL_DIR / "all_encoders.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_kyc, y_kyc, test_size=0.2, random_state=42, stratify=y_kyc)

kyc_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
kyc_model.fit(X_train, y_train)

y_pred = kyc_model.predict(X_test)
y_proba = kyc_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

joblib.dump(kyc_model, MODEL_DIR / "kyc_model.pkl")
joblib.dump(kyc_features, MODEL_DIR / "kyc_features.pkl")

metrics['kyc_aml'] = {
    'auc': round(auc, 4),
    'accuracy': round(report['accuracy'], 4),
    'precision': round(report['1']['precision'], 4),
    'recall': round(report['1']['recall'], 4),
    'f1': round(report['1']['f1-score'], 4),
    'confusion_matrix': cm.tolist(),
    'feature_count': len(kyc_features),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'alert_rate': float(y_kyc.mean()),
}
fi = pd.DataFrame({'feature': kyc_features,
                   'importance': kyc_model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
metrics['kyc_aml']['feature_importance'] = fi.to_dict('records')

print(f"   ✓ AUC: {auc:.4f} | Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f}")

# ============================================================
# 4. CHURN PREDICTION MODEL (Gradient Boosting)
# ============================================================
print("\n[4/4] Training Customer Churn Prediction Model...")
churn = pd.read_csv(DATA_DIR / "churn.csv")

churn_features = ['age', 'annual_income', 'months_on_book', 'num_products',
                  'income_log', 'avg_monthly_balance', 'digital_engagement_score',
                  'complaints_last_year', 'num_branch_visits', 'credit_card_usage',
                  'gender', 'segment', 'employment_type']

X_churn = churn[churn_features].copy()
y_churn = churn['churn']

for col in ['gender', 'segment', 'employment_type']:
    le = LabelEncoder()
    X_churn[col] = le.fit_transform(X_churn[col].astype(str))
    encoders[f'churn_{col}'] = le

joblib.dump(encoders, MODEL_DIR / "all_encoders.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn)

X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

churn_model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='auc', use_label_encoder=False)
churn_model.fit(X_train_bal, y_train_bal)

y_pred = churn_model.predict(X_test)
y_proba = churn_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

joblib.dump(churn_model, MODEL_DIR / "churn_model.pkl")
joblib.dump(churn_features, MODEL_DIR / "churn_features.pkl")

metrics['churn'] = {
    'auc': round(auc, 4),
    'accuracy': round(report['accuracy'], 4),
    'precision': round(report['1']['precision'], 4),
    'recall': round(report['1']['recall'], 4),
    'f1': round(report['1']['f1-score'], 4),
    'confusion_matrix': cm.tolist(),
    'feature_count': len(churn_features),
    'train_samples': len(X_train_bal),
    'test_samples': len(X_test),
    'churn_rate': float(y_churn.mean()),
}
fi = pd.DataFrame({'feature': churn_features,
                   'importance': churn_model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
metrics['churn']['feature_importance'] = fi.to_dict('records')

print(f"   ✓ AUC: {auc:.4f} | Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f}")

# ============================================================
# SAVE METRICS
# ============================================================
with open(MODEL_DIR / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save feature lists for app
feature_info = {
    'credit_features': credit_features,
    'fraud_features': fraud_features,
    'kyc_features': kyc_features,
    'churn_features': churn_features,
}
with open(MODEL_DIR / "feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
for name, m in metrics.items():
    print(f"  {name:20s} | AUC: {m['auc']:.4f} | F1: {m['f1']:.4f}")
print(f"\nModels saved to: {MODEL_DIR}")
print("=" * 60)
