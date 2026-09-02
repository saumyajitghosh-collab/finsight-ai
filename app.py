"""
FinSight AI — Banking Intelligence Platform (Production-Optimized)
Optimized for Render free tier (512MB RAM):
  - Lazy-loads DataFrames only when first API call is made
  - Loads CSVs with optimized dtypes to reduce memory
  - Loads models on-demand
"""
import os, json, gc
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ─── Lazy-loaded globals ────────────────────────────────────────
_credit_model = None
_fraud_model = None
_kyc_model = None
_churn_model = None
_encoders = None
_credit_features = None
_fraud_features = None
_kyc_features = None
_churn_features = None
_model_metrics = None
_asset_stats = None
customers_df = None
txns_df = None
loans_df = None
kyc_df = None
portfolios_df = None
churn_df = None


def _load_models():
    global _credit_model, _fraud_model, _kyc_model, _churn_model
    global _encoders, _credit_features, _fraud_features, _kyc_features, _churn_features
    global _model_metrics, _asset_stats
    if _credit_model is not None:
        return
    print("Loading models...")
    _credit_model = joblib.load(MODEL_DIR / "credit_risk_model.pkl")
    _fraud_model = joblib.load(MODEL_DIR / "fraud_model.pkl")
    _kyc_model = joblib.load(MODEL_DIR / "kyc_model.pkl")
    _churn_model = joblib.load(MODEL_DIR / "churn_model.pkl")
    _encoders = joblib.load(MODEL_DIR / "all_encoders.pkl")
    _credit_features = joblib.load(MODEL_DIR / "credit_features.pkl")
    _fraud_features = joblib.load(MODEL_DIR / "fraud_features.pkl")
    _kyc_features = joblib.load(MODEL_DIR / "kyc_features.pkl")
    _churn_features = joblib.load(MODEL_DIR / "churn_features.pkl")
    with open(MODEL_DIR / "model_metrics.json") as f:
        _model_metrics = json.load(f)
    with open(DATA_DIR / "asset_stats.json") as f:
        _asset_stats = json.load(f)
    print("Models loaded.")


def _load_data():
    global customers_df, txns_df, loans_df, kyc_df, portfolios_df, churn_df
    if customers_df is not None:
        return
    print("Loading data...")
    # Load with optimized dtypes to save memory
    customers_df = pd.read_csv(DATA_DIR / "customers.csv")
    txns_df = pd.read_csv(DATA_DIR / "transactions.csv", dtype={
        'txn_type': 'category', 'channel': 'category',
        'merchant': 'category', 'customer_id': 'string', 'txn_id': 'string'
    })
    loans_df = pd.read_csv(DATA_DIR / "loans.csv", dtype={
        'loan_type': 'category', 'loan_purpose': 'category',
        'employment_type': 'category', 'segment': 'category',
        'customer_id': 'string', 'loan_id': 'string'
    })
    kyc_df = pd.read_csv(DATA_DIR / "kyc_aml.csv", dtype={
        'entity_type': 'category', 'country': 'category',
        'doc_type': 'category', 'ownership_transparency': 'category',
        'risk_category': 'category', 'entity_id': 'string'
    })
    portfolios_df = pd.read_csv(DATA_DIR / "portfolios.csv")
    churn_df = pd.read_csv(DATA_DIR / "churn.csv", dtype={
        'gender': 'category', 'segment': 'category',
        'employment_type': 'category', 'customer_id': 'string'
    })
    gc.collect()
    print(f"Data loaded: {len(customers_df)} customers, {len(txns_df)} txns, {len(loans_df)} loans")


def encode_input(df, feature_list, encoder_prefix, encoders):
    df = df.copy()
    for col in feature_list:
        if col in df.columns:
            le_key = f"{encoder_prefix}_{col}"
            if le_key in encoders:
                le = encoders[le_key]
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    return df


# ─── Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/dashboard')
def dashboard():
    _load_models()
    _load_data()
    total_customers = len(customers_df)
    total_txns = len(txns_df)
    fraud_count = int(txns_df['is_fraud'].sum())
    total_loans = len(loans_df)
    default_count = int(loans_df['default'].sum())
    kyc_alerts = int(kyc_df['alert_flag'].sum())
    churn_count = int(churn_df['churn'].sum())
    return jsonify({
        'total_customers': total_customers,
        'total_transactions': total_txns,
        'fraud_count': fraud_count,
        'fraud_rate': round(fraud_count / total_txns * 100, 2),
        'total_loans': total_loans,
        'default_count': default_count,
        'default_rate': round(default_count / total_loans * 100, 2),
        'kyc_alerts': kyc_alerts,
        'churn_count': churn_count,
        'churn_rate': round(churn_count / len(churn_df) * 100, 2),
        'segment_distribution': customers_df['segment'].value_counts().to_dict(),
        'loan_type_distribution': loans_df['loan_type'].value_counts().to_dict(),
        'fraud_by_type': txns_df.groupby('txn_type')['is_fraud'].sum().to_dict(),
        'risk_category_distribution': kyc_df['risk_category'].value_counts().to_dict(),
        'model_metrics': _model_metrics,
    })


@app.route('/credit-risk')
def credit_risk_page():
    return render_template('credit_risk.html')


@app.route('/api/credit-risk/predict', methods=['POST'])
def credit_risk_predict():
    _load_models()
    data = request.json
    input_data = pd.DataFrame([{
        'loan_amount': float(data['loan_amount']),
        'interest_rate': float(data['interest_rate']),
        'tenure_months': int(data['tenure_months']),
        'age': int(data['age']),
        'annual_income': float(data['annual_income']),
        'months_on_book': int(data.get('months_on_book', 36)),
        'num_products': int(data.get('num_products', 2)),
        'loan_to_income': float(data['loan_amount']) / float(data['annual_income']),
        'emi_to_income': (float(data['loan_amount']) / int(data['tenure_months'])) / (float(data['annual_income']) / 12),
        'income_log': np.log1p(float(data['annual_income'])),
        'loan_amount_log': np.log1p(float(data['loan_amount'])),
        'credit_bureau_score': int(data['credit_bureau_score']),
        'employment_type': data['employment_type'],
        'loan_type': data['loan_type'],
        'segment': data.get('segment', 'Retail'),
    }])
    input_encoded = encode_input(input_data, _credit_features, 'credit', _encoders)
    input_encoded = input_encoded[_credit_features]
    proba = _credit_model.predict_proba(input_encoded)[0]
    default_prob = float(proba[1])
    prediction = int(default_prob > 0.5)
    if default_prob < 0.05:
        grade, recommendation = 'A', 'APPROVE — Low risk borrower'
    elif default_prob < 0.15:
        grade, recommendation = 'B', 'APPROVE — Moderate risk, standard terms'
    elif default_prob < 0.30:
        grade, recommendation = 'C', 'CONDITIONAL — Higher interest or collateral required'
    elif default_prob < 0.50:
        grade, recommendation = 'D', 'REVIEW — Manual underwriting recommended'
    else:
        grade, recommendation = 'E', 'DECLINE — High default probability'
    fi_list = _model_metrics['credit_risk']['feature_importance'][:5]
    return jsonify({
        'default_probability': round(default_prob * 100, 2),
        'prediction': 'DEFAULT' if prediction else 'NO DEFAULT',
        'risk_grade': grade,
        'recommendation': recommendation,
        'top_risk_factors': fi_list,
        'input_summary': {
            'loan_amount': f"₹{float(data['loan_amount']):,.0f}",
            'loan_to_income': f"{float(data['loan_amount'])/float(data['annual_income']):.2f}",
            'credit_score': data['credit_bureau_score'],
        }
    })


@app.route('/api/credit-risk/sample')
def credit_risk_sample():
    _load_data()
    sample = loans_df.sample(1).iloc[0]
    return jsonify({
        'loan_amount': int(sample['loan_amount']),
        'interest_rate': float(sample['interest_rate']),
        'tenure_months': int(sample['tenure_months']),
        'age': int(sample['age']),
        'annual_income': int(sample['annual_income']),
        'credit_bureau_score': int(sample['credit_bureau_score']),
        'employment_type': str(sample['employment_type']),
        'loan_type': str(sample['loan_type']),
        'segment': str(sample['segment']),
        'actual_default': int(sample['default']),
    })


@app.route('/fraud-detection')
def fraud_page():
    return render_template('fraud_detection.html')


@app.route('/api/fraud/predict', methods=['POST'])
def fraud_predict():
    _load_models()
    data = request.json
    input_data = pd.DataFrame([{
        'amount': float(data['amount']),
        'amount_log': np.log1p(float(data['amount'])),
        'txn_type': data['txn_type'],
        'channel': data['channel'],
        'merchant': data['merchant'],
        'hour_of_day': int(data['hour_of_day']),
        'day_of_week': int(data['day_of_week']),
        'is_international': int(data.get('is_international', 0)),
        'is_night_txn': int(1 if (int(data['hour_of_day']) < 6 or int(data['hour_of_day']) > 22) else 0),
        'is_weekend': int(1 if int(data['day_of_week']) >= 5 else 0),
        'is_high_amount': int(1 if float(data['amount']) > 50000 else 0),
    }])
    input_encoded = encode_input(input_data, _fraud_features, 'fraud', _encoders)
    input_encoded = input_encoded[_fraud_features]
    proba = _fraud_model.predict_proba(input_encoded)[0]
    fraud_prob = float(proba[1])
    prediction = int(fraud_prob > 0.5)
    if fraud_prob > 0.7:
        action = 'BLOCK TRANSACTION — High fraud probability'
    elif fraud_prob > 0.4:
        action = 'STEP-UP AUTH — Require OTP/biometric verification'
    elif fraud_prob > 0.15:
        action = 'MONITOR — Flag for review, allow transaction'
    else:
        action = 'APPROVE — Low risk transaction'
    return jsonify({
        'fraud_probability': round(fraud_prob * 100, 2),
        'prediction': 'FRAUD' if prediction else 'LEGITIMATE',
        'action': action,
        'risk_indicators': [
            f"Night transaction ({'Yes' if int(data['hour_of_day']) < 6 or int(data['hour_of_day']) > 22 else 'No'})",
            f"International ({'Yes' if int(data.get('is_international', 0)) else 'No'})",
            f"High amount >₹50k ({'Yes' if float(data['amount']) > 50000 else 'No'})",
            f"Weekend ({'Yes' if int(data['day_of_week']) >= 5 else 'No'})",
        ]
    })


@app.route('/api/fraud/sample')
def fraud_sample():
    _load_data()
    sample = txns_df.sample(1).iloc[0]
    return jsonify({
        'amount': float(sample['amount']),
        'txn_type': str(sample['txn_type']),
        'channel': str(sample['channel']),
        'merchant': str(sample['merchant']),
        'hour_of_day': int(sample['hour_of_day']),
        'day_of_week': int(sample['day_of_week']),
        'is_international': int(sample['is_international']),
        'actual_fraud': int(sample['is_fraud']),
    })


@app.route('/kyc-aml')
def kyc_page():
    return render_template('kyc_aml.html')


@app.route('/api/kyc/predict', methods=['POST'])
def kyc_predict():
    _load_models()
    data = request.json
    risk_countries = {'Cyprus', 'BVI', 'Panama', 'Seychelles', 'Mauritius', 'Cayman Islands'}
    input_data = pd.DataFrame([{
        'entity_type': data['entity_type'],
        'country': data['country'],
        'doc_type': data['doc_type'],
        'doc_verified': int(data['doc_verified']),
        'num_bank_accounts': int(data['num_bank_accounts']),
        'txn_volume_monthly': float(data['txn_volume_monthly']),
        'has_pep_link': int(data.get('has_pep_link', 0)),
        'has_sanction_link': int(data.get('has_sanction_link', 0)),
        'adverse_media_hits': int(data.get('adverse_media_hits', 0)),
        'years_in_business': int(data['years_in_business']),
        'ownership_transparency': data['ownership_transparency'],
        'txn_volume_log': np.log1p(float(data['txn_volume_monthly'])),
        'high_risk_country': int(1 if data['country'] in risk_countries else 0),
    }])
    input_encoded = encode_input(input_data, _kyc_features, 'kyc', _encoders)
    input_encoded = input_encoded[_kyc_features]
    proba = _kyc_model.predict_proba(input_encoded)[0]
    alert_prob = float(proba[1])
    prediction = int(alert_prob > 0.5)
    risk_score = (
        int(data.get('has_sanction_link', 0)) * 40 +
        int(data.get('has_pep_link', 0)) * 15 +
        int(data.get('adverse_media_hits', 0)) * 8 +
        (1 if data['country'] in risk_countries else 0) * 12 +
        (1 if data['ownership_transparency'] == 'Low' else 0) * 6 +
        (1 if int(data['doc_verified']) == 0 else 0) * 5 +
        max(0, np.log1p(float(data['txn_volume_monthly'])) - 12) * 0.3
    )
    risk_score = min(100, int(risk_score))
    if risk_score <= 20: category = 'Low'
    elif risk_score <= 40: category = 'Medium'
    elif risk_score <= 60: category = 'High'
    else: category = 'Critical'
    if category in ('High', 'Critical') or int(data.get('has_sanction_link', 0)):
        action = 'ESCALATE — Enhanced due diligence required, freeze onboarding'
    elif category == 'Medium':
        action = 'REVIEW — Additional documentation requested'
    else:
        action = 'CLEAR — Standard onboarding approved'
    return jsonify({
        'risk_score': risk_score,
        'risk_category': category,
        'alert_probability': round(alert_prob * 100, 2),
        'alert': 'YES' if prediction else 'NO',
        'action': action,
        'risk_factors': [
            f"Sanction link: {'YES' if int(data.get('has_sanction_link', 0)) else 'No'}",
            f"PEP link: {'YES' if int(data.get('has_pep_link', 0)) else 'No'}",
            f"High-risk country: {'YES' if data['country'] in risk_countries else 'No'}",
            f"Adverse media hits: {int(data.get('adverse_media_hits', 0))}",
            f"Ownership transparency: {data['ownership_transparency']}",
            f"Document verified: {'No' if int(data['doc_verified']) == 0 else 'Yes'}",
        ]
    })


@app.route('/api/kyc/sample')
def kyc_sample():
    _load_data()
    sample = kyc_df.sample(1).iloc[0]
    return jsonify({
        'entity_type': str(sample['entity_type']),
        'country': str(sample['country']),
        'doc_type': str(sample['doc_type']),
        'doc_verified': int(sample['doc_verified']),
        'num_bank_accounts': int(sample['num_bank_accounts']),
        'txn_volume_monthly': float(sample['txn_volume_monthly']),
        'has_pep_link': int(sample['has_pep_link']),
        'has_sanction_link': int(sample['has_sanction_link']),
        'adverse_media_hits': int(sample['adverse_media_hits']),
        'years_in_business': int(sample['years_in_business']),
        'ownership_transparency': str(sample['ownership_transparency']),
        'actual_alert': int(sample['alert_flag']),
    })


@app.route('/portfolio-advisor')
def portfolio_page():
    return render_template('portfolio_advisor.html')


@app.route('/api/portfolio/optimize', methods=['POST'])
def portfolio_optimize():
    _load_models()
    data = request.json
    risk_profile = data['risk_profile']
    horizon = int(data.get('investment_horizon', 5))
    capital = float(data.get('capital', 1000000))
    allocations = {
        'Conservative': {
            'Equity_LargeCap': 0.15, 'Equity_MidCap': 0.05, 'Equity_SmallCap': 0.00,
            'Debt_Govt': 0.30, 'Debt_Corporate': 0.20, 'Gold': 0.10,
            'International': 0.05, 'REIT': 0.05, 'Commodities': 0.03, 'Cash': 0.07,
        },
        'Moderate': {
            'Equity_LargeCap': 0.25, 'Equity_MidCap': 0.10, 'Equity_SmallCap': 0.05,
            'Debt_Govt': 0.20, 'Debt_Corporate': 0.15, 'Gold': 0.08,
            'International': 0.07, 'REIT': 0.05, 'Commodities': 0.03, 'Cash': 0.02,
        },
        'Aggressive': {
            'Equity_LargeCap': 0.30, 'Equity_MidCap': 0.15, 'Equity_SmallCap': 0.10,
            'Debt_Govt': 0.10, 'Debt_Corporate': 0.10, 'Gold': 0.05,
            'International': 0.10, 'REIT': 0.05, 'Commodities': 0.04, 'Cash': 0.01,
        },
        'Very Aggressive': {
            'Equity_LargeCap': 0.25, 'Equity_MidCap': 0.20, 'Equity_SmallCap': 0.15,
            'Debt_Govt': 0.05, 'Debt_Corporate': 0.05, 'Gold': 0.05,
            'International': 0.15, 'REIT': 0.05, 'Commodities': 0.04, 'Cash': 0.01,
        },
    }
    alloc = allocations[risk_profile]
    if horizon > 10:
        for eq in ['Equity_LargeCap', 'Equity_MidCap', 'Equity_SmallCap']:
            alloc[eq] = round(alloc[eq] * 1.15, 4)
        total = sum(alloc.values())
        alloc = {k: round(v/total, 4) for k, v in alloc.items()}
    ret = sum(alloc[a] * _asset_stats[a][0] for a in alloc)
    vol = np.sqrt(sum((alloc[a] * _asset_stats[a][1])**2 for a in alloc) +
                  2 * sum(alloc[a] * alloc[b] * _asset_stats[a][1] * _asset_stats[b][1] * 0.3
                          for i, a in enumerate(alloc) for b in list(alloc)[i+1:]))
    sharpe = (ret - 0.035) / vol if vol > 0 else 0
    allocation_amounts = {k: round(v * capital) for k, v in alloc.items()}
    return jsonify({
        'allocation': alloc,
        'allocation_amounts': allocation_amounts,
        'expected_return': round(ret * 100, 2),
        'expected_volatility': round(vol * 100, 2),
        'sharpe_ratio': round(sharpe, 3),
        'risk_profile': risk_profile,
        'investment_horizon': horizon,
        'capital': capital,
    })


@app.route('/churn')
def churn_page():
    return render_template('churn.html')


@app.route('/api/churn/predict', methods=['POST'])
def churn_predict():
    _load_models()
    data = request.json
    input_data = pd.DataFrame([{
        'age': int(data['age']),
        'annual_income': float(data['annual_income']),
        'months_on_book': int(data['months_on_book']),
        'num_products': int(data['num_products']),
        'income_log': np.log1p(float(data['annual_income'])),
        'avg_monthly_balance': float(data['avg_monthly_balance']),
        'digital_engagement_score': int(data['digital_engagement_score']),
        'complaints_last_year': int(data['complaints_last_year']),
        'num_branch_visits': int(data['num_branch_visits']),
        'credit_card_usage': int(data['credit_card_usage']),
        'gender': data.get('gender', 'M'),
        'segment': data.get('segment', 'Retail'),
        'employment_type': data.get('employment_type', 'Salaried'),
    }])
    input_encoded = encode_input(input_data, _churn_features, 'churn', _encoders)
    input_encoded = input_encoded[_churn_features]
    proba = _churn_model.predict_proba(input_encoded)[0]
    churn_prob = float(proba[1])
    prediction = int(churn_prob > 0.5)
    if churn_prob > 0.5:
        action = 'RETENTION CAMPAIGN — High churn risk, immediate intervention needed'
    elif churn_prob > 0.25:
        action = 'PROACTIVE OUTREACH — Offer personalized benefits or fee waivers'
    elif churn_prob > 0.10:
        action = 'MONITOR — Include in watchlist, increase engagement touchpoints'
    else:
        action = 'LOW RISK — No action needed'
    return jsonify({
        'churn_probability': round(churn_prob * 100, 2),
        'prediction': 'CHURN' if prediction else 'RETAIN',
        'action': action,
        'risk_factors': [
            f"Digital engagement: {data['digital_engagement_score']}/100",
            f"Complaints (last year): {data['complaints_last_year']}",
            f"Products held: {data['num_products']}",
            f"Branch visits: {data['num_branch_visits']}",
        ]
    })


@app.route('/api/churn/sample')
def churn_sample():
    _load_data()
    sample = churn_df.sample(1).iloc[0]
    return jsonify({
        'age': int(sample['age']),
        'annual_income': int(sample['annual_income']),
        'months_on_book': int(sample['months_on_book']),
        'num_products': int(sample['num_products']),
        'avg_monthly_balance': int(sample['avg_monthly_balance']),
        'digital_engagement_score': int(sample['digital_engagement_score']),
        'complaints_last_year': int(sample['complaints_last_year']),
        'num_branch_visits': int(sample['num_branch_visits']),
        'credit_card_usage': int(sample['credit_card_usage']),
        'gender': str(sample['gender']),
        'segment': str(sample['segment']),
        'employment_type': str(sample['employment_type']),
        'actual_churn': int(sample['churn']),
    })


@app.route('/models')
def models_page():
    _load_models()
    return render_template('models.html', metrics=_model_metrics)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
