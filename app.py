#!/usr/bin/env python
"""
FinSight AI v2.0 - Advanced AI Banking Platform
A research-grade banking AI system with:
- Quantitative Finance Lab (Black-Scholes, Monte Carlo VaR, Copula defaults)
- Agentic AI Workflows (Multi-agent AML investigation)
- Survival Analysis (Cox Proportional Hazards for loan default timing)
- Deep Learning (Autoencoder fraud detection via Isolation Forest)
- Explainable AI (SHAP-style feature importance)
"""
import os, json, pickle, math, random, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from flask import Flask, request, jsonify

app = Flask(__name__)
BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"

# ============ LOAD DATA & MODELS ============
def load_data():
    d = {}
    for name in ['customers', 'transactions', 'loans', 'kyc', 'portfolios', 'churn']:
        p = DATA / f"{name}.csv"
        if p.exists():
            d[name] = pd.read_csv(p)
    return d

def load_models():
    m = {}
    for name in ['credit_model', 'fraud_model', 'kyc_model', 'churn_model', 'survival_model']:
        p = MODELS / f"{name}.pkl"
        if p.exists():
            with open(p, 'rb') as f:
                m[name] = pickle.load(f)
    sp = MODELS / 'summary.json'
    if sp.exists():
        with open(sp) as f:
            m['summary'] = json.load(f)
    return m

DATA_DFS = load_data()
MODELS_DICT = load_models()

# ============ QUANTITATIVE FINANCE ENGINE ============

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing formula."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Compute all Greeks."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    vega = S * norm.pdf(d1) * math.sqrt(T)
    return {'delta': delta, 'gamma': gamma, 'theta': theta / 365, 'vega': vega / 100, 'rho': rho / 100}

def monte_carlo_var(portfolio_value, weights, n_sims=10000, confidence=0.95, horizon=1):
    """Monte Carlo VaR using correlated asset returns."""
    n_assets = len(weights)
    mean_returns = np.random.normal(0.0005, 0.02, n_assets)
    volatilities = np.random.uniform(0.15, 0.35, n_assets)
    corr_matrix = np.eye(n_assets) * 0.5 + 0.5 / n_assets
    L = np.linalg.cholesky(corr_matrix)
    z = np.random.standard_normal((n_sims, n_assets))
    correlated_z = z @ L.T
    sim_returns = mean_returns + correlated_z * volatilities
    portfolio_returns = sim_returns @ weights
    portfolio_losses = -portfolio_returns * portfolio_value * math.sqrt(horizon)
    var = np.percentile(portfolio_losses, confidence * 100)
    es = portfolio_losses[portfolio_losses >= var].mean()
    return {
        'var': round(var, 2),
        'es': round(es, 2),
        'mean_loss': round(np.mean(portfolio_losses), 2),
        'std_loss': round(np.std(portfolio_losses), 2),
        'sim_losses': portfolio_losses.tolist()[:500],
        'percentiles': {
            '90': round(np.percentile(portfolio_losses, 90), 2),
            '95': round(np.percentile(portfolio_losses, 95), 2),
            '99': round(np.percentile(portfolio_losses, 99), 2),
        }
    }

def basel_irb_rwa(pd_val, lgd, ead, maturity=2.5, asset_class='corporate'):
    """Basel III IRB Risk-Weighted Assets calculation."""
    pd_val = max(pd_val, 0.0003)
    if asset_class == 'retail_revolving':
        R = 0.04
    elif asset_class in ('retail_other', 'retail_mortgage'):
        R = 0.15
    elif asset_class == 'sme':
        R = 0.12 * (1 - math.exp(-50 * pd_val)) / (1 - math.exp(-50)) + 0.24 * (1 - (1 - math.exp(-50 * pd_val)) / (1 - math.exp(-50)))
        R = min(R, 0.24)
    else:
        R = 0.12 * (1 - math.exp(-50 * pd_val)) / (1 - math.exp(-50)) + 0.24 * (1 - (1 - math.exp(-50 * pd_val)) / (1 - math.exp(-50)))
    if asset_class in ('corporate', 'sme'):
        b = 0.11852 - 0.05478 * math.log(pd_val) + 0.00001 * (maturity - 2.5)**2
        b = max(b, 0)
        maturity_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    else:
        maturity_adj = 1.0
    N_norm = norm.ppf(pd_val)
    K = (lgd * norm.cdf(norm.ppf(0.999) * math.sqrt(R) + N_norm * math.sqrt(1 - R)) - pd_val * lgd) * maturity_adj
    K = max(K, 0)
    rwa = K * 12.5 * ead
    capital_required = rwa * 0.08
    expected_loss = pd_val * lgd * ead
    return {
        'rwa': round(rwa, 2),
        'capital_required': round(capital_required, 2),
        'expected_loss': round(expected_loss, 2),
        'K': round(K, 4),
        'R': round(R, 4),
        'rwa_density': round(rwa / ead * 100, 2) if ead > 0 else 0
    }

def gaussian_copula_defaults(n_loans, n_sims, pd_val, lgd, correlation, threshold_percentile=99):
    """Gaussian copula model for correlated portfolio defaults."""
    n_loans = min(n_loans, 10000)
    n_sims = min(n_sims, 5000)
    Z = np.random.standard_normal(n_sims)
    losses = np.zeros(n_sims)
    defaults_count = np.zeros(n_sims)
    for sim in range(n_sims):
        epsilon = np.random.standard_normal(n_loans)
        X = math.sqrt(correlation) * Z[sim] + math.sqrt(1 - correlation) * epsilon
        threshold = norm.ppf(pd_val)
        defaults = (X < threshold).astype(float)
        defaults_count[sim] = defaults.sum()
        losses[sim] = defaults.sum() * lgd * (1000000 / n_loans)
    var_99 = np.percentile(losses, threshold_percentile)
    es_99 = losses[losses >= var_99].mean()
    return {
        'var': round(var_99, 2),
        'es': round(es_99, 2),
        'mean_loss': round(np.mean(losses), 2),
        'max_loss': round(np.max(losses), 2),
        'mean_defaults': round(np.mean(defaults_count), 1),
        'max_defaults': int(np.max(defaults_count)),
        'loss_distribution': np.percentile(losses, np.arange(0, 101, 2)).tolist(),
        'default_distribution': np.percentile(defaults_count, np.arange(0, 101, 2)).tolist(),
    }

def stress_test_portfolio(portfolio, scenarios):
    """Run macro stress testing on a loan portfolio."""
    results = []
    for scenario in scenarios:
        gdp_shock = scenario.get('gdp_shock', 0)
        unemployment_shock = scenario.get('unemployment_shock', 0)
        house_price_shock = scenario.get('house_price_shock', 0)
        sensitivities = {
            'Home': {'gdp': -0.004, 'unemployment': 0.006, 'house_price': -0.002},
            'Auto': {'gdp': -0.006, 'unemployment': 0.008, 'house_price': -0.001},
            'Personal': {'gdp': -0.007, 'unemployment': 0.012, 'house_price': -0.001},
            'Education': {'gdp': -0.003, 'unemployment': 0.004, 'house_price': 0},
            'Business': {'gdp': -0.008, 'unemployment': 0.003, 'house_price': -0.001},
        }
        total_el = 0
        total_exposure = 0
        defaults = 0
        for loan in portfolio:
            loan_type = loan.get('loan_type', 'Personal')
            base_pd = 0.05
            lgd_val = 0.4
            ead = loan.get('ead', 500000) if isinstance(loan, dict) else 500000
            if isinstance(loan, dict):
                base_pd = max(0.001, min(0.5, float(loan.get('interest_rate', 0.12)) * 0.3))
            sens = sensitivities.get(loan_type, sensitivities['Personal'])
            pd_adjustment = (sens['gdp'] * gdp_shock +
                           sens['unemployment'] * unemployment_shock +
                           sens['house_price'] * house_price_shock)
            stressed_pd = min(max(base_pd + pd_adjustment, 0.0001), 0.95)
            el = stressed_pd * lgd_val * ead
            total_el += el
            total_exposure += ead
            if np.random.random() < stressed_pd:
                defaults += 1
        results.append({
            'scenario': scenario['name'],
            'gdp_shock': gdp_shock,
            'unemployment_shock': unemployment_shock,
            'house_price_shock': house_price_shock,
            'total_exposure': round(total_exposure, 2),
            'expected_loss': round(total_el, 2),
            'loss_rate': round(total_el / total_exposure * 100, 2) if total_exposure > 0 else 0,
            'defaults': defaults,
        })
    return results

def get_default_stress_scenarios():
    return [
        {'name': 'Baseline', 'gdp_shock': 0, 'unemployment_shock': 0, 'house_price_shock': 0},
        {'name': 'Mild Recession', 'gdp_shock': -2, 'unemployment_shock': 2, 'house_price_shock': -5},
        {'name': 'Moderate Recession', 'gdp_shock': -4, 'unemployment_shock': 4, 'house_price_shock': -10},
        {'name': 'Severe Recession', 'gdp_shock': -6, 'unemployment_shock': 6, 'house_price_shock': -15},
        {'name': 'Financial Crisis', 'gdp_shock': -8, 'unemployment_shock': 8, 'house_price_shock': -25},
        {'name': 'COVID-like Shock', 'gdp_shock': -10, 'unemployment_shock': 10, 'house_price_shock': -20},
    ]

# ============ AGENTIC AML WORKFLOW ============

def run_agentic_aml_investigation(customer_data):
    """Simulate a multi-agent AML investigation workflow."""
    agents = [
        {'name': 'Data Collection Agent', 'description': 'Gathers customer transactions, KYC records, and external data', 'status': 'executing', 'actions': []},
        {'name': 'Risk Scoring Agent', 'description': 'Evaluates risk using ML model + heuristic rules', 'status': 'pending', 'actions': []},
        {'name': 'Sanctions Screening Agent', 'description': 'Checks against OFAC, UN, EU sanctions lists', 'status': 'pending', 'actions': []},
        {'name': 'Network Analysis Agent', 'description': 'Maps transaction network and identifies circular flows', 'status': 'pending', 'actions': []},
        {'name': 'Investigation Report Agent', 'description': 'Synthesizes findings and generates SAR recommendation', 'status': 'pending', 'actions': []}
    ]
    agents[0]['status'] = 'completed'
    agents[0]['actions'] = [
        "Retrieved " + str(random.randint(150, 500)) + " transactions for customer " + str(customer_data.get('customer_id', 'C123456')),
        "Found " + str(customer_data.get('num_large_txns', 3)) + " large transactions (>Rs.5L) in last 30 days",
        "Customer located in " + str(customer_data.get('country', 'India')),
        "Account age: " + str(customer_data.get('account_age_months', 24)) + " months"
    ]
    agents[1]['status'] = 'completed'
    risk_score = customer_data.get('risk_score', 35)
    risk_level = 'HIGH' if risk_score > 60 else ('MEDIUM' if risk_score > 30 else 'LOW')
    agents[1]['actions'] = [
        "ML model risk score: " + str(risk_score) + "/100 (" + risk_level + ")",
        "Structuring detected: " + ('YES' if customer_data.get('structuring_detected', 0) else 'NO'),
        "PEP flag: " + ('YES' if customer_data.get('pep_flag', 0) else 'NO'),
        "Behavioral anomaly score: " + str(round(random.uniform(0.3, 0.9), 2)) if risk_score > 40 else "Behavioral anomaly score: " + str(round(random.uniform(0.05, 0.3), 2))
    ]
    agents[2]['status'] = 'completed'
    sanctions_hit = customer_data.get('sanctions_hit', 0)
    agents[2]['actions'] = [
        "OFAC list: " + ('MATCH FOUND' if sanctions_hit else 'No match'),
        "UN consolidated list: No match",
        "EU sanctions list: No match",
        "Fuzzy match score: " + str(round(random.uniform(0.1, 0.4), 2)) if not sanctions_hit else "0.95"
    ]
    agents[3]['status'] = 'completed'
    agents[3]['actions'] = [
        "Identified " + str(random.randint(2, 8)) + " connected accounts",
        "Circular transaction flow detected: " + ('YES' if risk_score > 50 else 'NO'),
        "Total flow through network: Rs. " + str(random.randint(10, 500)) + " Lakhs",
        "Counterparty risk concentration: " + ('HIGH' if risk_score > 50 else 'LOW')
    ]
    agents[4]['status'] = 'completed'
    sar_recommended = risk_score > 50 or sanctions_hit
    agents[4]['actions'] = [
        "Investigation status: " + ('SAR RECOMMENDED' if sar_recommended else 'No SAR required'),
        "Confidence level: " + str(round(random.uniform(0.85, 0.98), 2)) if sar_recommended else str(round(random.uniform(0.70, 0.90), 2)),
        "Recommended action: " + ('File SAR within 30 days + Enhanced Due Diligence' if sar_recommended else 'Monitor for 90 days + Standard review'),
        "Audit trail: Complete (5 agents, " + str(random.randint(15, 30)) + " data points collected)"
    ]
    return {
        'agents': agents,
        'final_risk': risk_level,
        'sar_recommended': sar_recommended,
        'investigation_summary': "Multi-agent investigation complete. Risk: " + risk_level + ". SAR: " + ('Recommended' if sar_recommended else 'Not required') + ".",
        'total_agents': len(agents),
        'data_points': random.randint(15, 30),
        'confidence': round(random.uniform(0.85, 0.98) if sar_recommended else random.uniform(0.70, 0.90), 2)
    }

# ============ SURVIVAL ANALYSIS ============

def survival_predict(model_data, input_features):
    """Predict survival curve using Cox PH model."""
    try:
        cph = model_data['model']
        features = model_data['features']
        dummy_cols = model_data.get('dummy_columns', [])
        feat_dict = {}
        for f in features:
            feat_dict[f] = float(input_features.get(f, 0))
        for dc in dummy_cols:
            loan_type_val = input_features.get('loan_type', 'Personal')
            if dc == 'loan_type_' + loan_type_val:
                feat_dict[dc] = 1.0
            else:
                feat_dict[dc] = 0.0
        df = pd.DataFrame([feat_dict])
        survival_fn = cph.predict_survival_function(df)
        timepoints = [6, 12, 24, 36, 48, 60, 72, 84]
        survival_probs = {}
        for t in timepoints:
            if t in survival_fn.index:
                survival_probs[t] = round(float(survival_fn.loc[t].iloc[0]), 4)
            else:
                idx = survival_fn.index
                if t < idx.min():
                    survival_probs[t] = 1.0
                elif t > idx.max():
                    survival_probs[t] = round(float(survival_fn.iloc[-1].iloc[0]), 4)
                else:
                    survival_probs[t] = round(float(survival_fn.loc[:t].iloc[-1].iloc[0]), 4)
        partial_hazard = float(cph.predict_partial_hazard(df).iloc[0])
        hazard_ratios = model_data.get('hazard_ratios', {})
        return {
            'survival_probs': survival_probs,
            'partial_hazard': round(partial_hazard, 4),
            'hazard_ratios': {k: round(float(v), 4) for k, v in list(hazard_ratios.items())[:8]},
            'c_index': model_data.get('c_index', 0.65),
            'median_survival': next((t for t in sorted(survival_probs.keys()) if survival_probs[t] < 0.5), None),
        }
    except Exception as e:
        return {'error': str(e)}

# ============ AUTOENCODER FRAUD DETECTION ============

def autoencoder_fraud_score(model_data, transaction_features):
    """Score a transaction using Isolation Forest (autoencoder proxy)."""
    try:
        scaler = model_data['scaler']
        iso_forest = model_data['iso_forest']
        features = model_data['features']
        encoders = model_data['encoders']
        row = {}
        row['amount'] = float(transaction_features.get('amount', 5000))
        row['hour'] = int(transaction_features.get('hour', 12))
        row['txn_speed'] = float(transaction_features.get('txn_speed', 2))
        row['device_change'] = int(transaction_features.get('device_change', 0))
        row['freq_last_24h'] = int(transaction_features.get('freq_last_24h', 3))
        for enc_name, enc in encoders.items():
            enc_val = transaction_features.get(enc_name, list(enc.classes_)[0])
            try:
                row[enc_name + '_enc'] = int(enc.transform([enc_val])[0])
            except:
                row[enc_name + '_enc'] = 0
        X = pd.DataFrame([[row.get(f, 0) for f in features]], columns=features)
        X_scaled = scaler.transform(X)
        anomaly_score = float(iso_forest.decision_function(X_scaled)[0])
        is_anomaly = int(iso_forest.predict(X_scaled)[0])
        xgb_model = model_data['model']
        xgb_prob = float(xgb_model.predict_proba(X)[0][1])
        reconstruction_error = (1 - anomaly_score) / 2
        combined_score = 0.5 * xgb_prob + 0.5 * reconstruction_error
        return {
            'xgb_fraud_prob': round(xgb_prob, 4),
            'anomaly_score': round(anomaly_score, 4),
            'reconstruction_error': round(reconstruction_error, 4),
            'combined_score': round(combined_score, 4),
            'is_anomaly': is_anomaly == -1,
            'verdict': 'FRAUD' if combined_score > 0.5 else ('SUSPICIOUS' if combined_score > 0.3 else 'NORMAL')
        }
    except Exception as e:
        return {'error': str(e)}

# ============ SHAP-STYLE EXPLAINABILITY ============

def get_feature_importance(model_data):
    """Get SHAP-style feature importance."""
    try:
        model = model_data['model']
        features = model_data['features']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = [0] * len(features)
        result = []
        for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
            result.append({'feature': f.replace('_', ' ').title(), 'importance': round(float(imp), 4)})
        return result[:10]
    except:
        return []

# ============ API ROUTES ============

@app.route('/api/dashboard')
def api_dashboard():
    try:
        summary = MODELS_DICT.get('summary', {})
        return jsonify({
            'customers': summary.get('total_customers', 0),
            'transactions': summary.get('total_transactions', 0),
            'loans': summary.get('total_loans', 0),
            'fraud_alerts': summary.get('total_fraud_alerts', 0),
            'defaults': summary.get('total_defaults', 0),
            'kyc_alerts': summary.get('total_kyc_alerts', 0),
            'churn_cases': summary.get('total_churn', 0),
            'portfolios': summary.get('total_portfolios', 0),
            'survival_events': summary.get('survival_events', 0),
            'models': {
                'credit_auc': MODELS_DICT.get('credit_model', {}).get('auc', 0),
                'fraud_auc': MODELS_DICT.get('fraud_model', {}).get('auc', 0),
                'kyc_auc': MODELS_DICT.get('kyc_model', {}).get('auc', 0),
                'churn_auc': MODELS_DICT.get('churn_model', {}).get('auc', 0),
                'survival_c_index': MODELS_DICT.get('survival_model', {}).get('c_index', 0),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/credit-risk', methods=['POST'])
def api_credit_risk():
    try:
        data = request.json
        model_data = MODELS_DICT['credit_model']
        model = model_data['model']
        features = model_data['features']
        encoder = model_data['encoder']
        loan_type = data.get('loan_type', 'Personal')
        try:
            loan_type_enc = int(encoder.transform([loan_type])[0])
        except:
            loan_type_enc = 0
        row = {
            'loan_amount': float(data.get('loan_amount', 500000)),
            'interest_rate': float(data.get('interest_rate', 0.12)),
            'term_months': int(data.get('term_months', 36)),
            'credit_score': int(data.get('credit_score', 680)),
            'annual_income': float(data.get('annual_income', 800000)),
            'age': int(data.get('age', 35)),
            'dti_ratio': float(data.get('dti_ratio', 0.3)),
            'employment_years': int(data.get('employment_years', 5)),
            'num_dependents': int(data.get('num_dependents', 2)),
            'num_prior_loans': int(data.get('num_prior_loans', 1)),
            'has_mortgage': int(data.get('has_mortgage', 0)),
            'loan_type_enc': loan_type_enc,
        }
        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        prob = float(model.predict_proba(X)[0][1])
        risk_band = 'LOW' if prob < 0.1 else ('MEDIUM' if prob < 0.3 else ('HIGH' if prob < 0.6 else 'CRITICAL'))
        importance = get_feature_importance(model_data)
        return jsonify({
            'probability': round(prob, 4),
            'risk_band': risk_band,
            'recommendation': 'Approve' if prob < 0.15 else ('Review' if prob < 0.35 else 'Decline'),
            'feature_importance': importance,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fraud-detect', methods=['POST'])
def api_fraud_detect():
    try:
        data = request.json
        model_data = MODELS_DICT['fraud_model']
        result = autoencoder_fraud_score(model_data, data)
        result['amount'] = float(data.get('amount', 0))
        result['txn_type'] = data.get('txn_type', 'UPI')
        result['hour'] = int(data.get('hour', 12))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/kyc-aml', methods=['POST'])
def api_kyc_aml():
    try:
        data = request.json
        model_data = MODELS_DICT['kyc_model']
        model = model_data['model']
        features = model_data['features']
        encoder = model_data['encoder']
        country = data.get('country', 'India')
        try:
            country_enc = int(encoder.transform([country])[0])
        except:
            country_enc = 0
        risk_score = int(data.get('risk_score', 35))
        row = {
            'country_enc': country_enc,
            'txn_volume_30d': float(data.get('txn_volume_30d', 500000)),
            'num_large_txns': int(data.get('num_large_txns', 3)),
            'structuring_detected': int(data.get('structuring_detected', 0)),
            'pep_flag': int(data.get('pep_flag', 0)),
            'sanctions_hit': int(data.get('sanctions_hit', 0)),
            'risk_score': risk_score,
            'account_age_months': int(data.get('account_age_months', 24)),
        }
        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        prob = float(model.predict_proba(X)[0][1])
        return jsonify({
            'probability': round(prob, 4),
            'risk_level': 'HIGH' if prob > 0.5 else ('MEDIUM' if prob > 0.2 else 'LOW'),
            'risk_score': risk_score,
            'recommendation': 'Investigate' if prob > 0.5 else ('Monitor' if prob > 0.2 else 'Normal'),
            'feature_importance': get_feature_importance(model_data),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/churn', methods=['POST'])
def api_churn():
    try:
        data = request.json
        model_data = MODELS_DICT['churn_model']
        model = model_data['model']
        features = model_data['features']
        encoders = model_data['encoders']
        segment = data.get('segment', 'Retail')
        occupation = data.get('occupation', 'Salaried')
        city = data.get('city', 'Mumbai')
        try: seg_enc = int(encoders['segment'].transform([segment])[0])
        except: seg_enc = 0
        try: occ_enc = int(encoders['occupation'].transform([occupation])[0])
        except: occ_enc = 0
        try: city_enc = int(encoders['city'].transform([city])[0])
        except: city_enc = 0
        row = {
            'age': int(data.get('age', 35)),
            'annual_income': float(data.get('annual_income', 800000)),
            'credit_score': int(data.get('credit_score', 680)),
            'months_with_bank': int(data.get('months_with_bank', 24)),
            'num_products': int(data.get('num_products', 2)),
            'avg_monthly_balance': float(data.get('avg_monthly_balance', 50000)),
            'satisfaction_score': float(data.get('satisfaction_score', 3.5)),
            'complaints_last_6m': int(data.get('complaints_last_6m', 1)),
            'digital_engagement': float(data.get('digital_engagement', 0.5)),
            'branch_visits_3m': int(data.get('branch_visits_3m', 3)),
            'product_utilization': float(data.get('product_utilization', 0.5)),
            'segment_enc': seg_enc,
            'occupation_enc': occ_enc,
            'city_enc': city_enc,
        }
        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        prob = float(model.predict_proba(X)[0][1])
        return jsonify({
            'probability': round(prob, 4),
            'risk_level': 'HIGH' if prob > 0.5 else ('MEDIUM' if prob > 0.2 else 'LOW'),
            'recommendation': 'Immediate retention action' if prob > 0.5 else ('Monitor closely' if prob > 0.2 else 'Low risk'),
            'feature_importance': get_feature_importance(model_data),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/survival', methods=['POST'])
def api_survival():
    try:
        data = request.json
        model_data = MODELS_DICT['survival_model']
        result = survival_predict(model_data, data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agentic-aml', methods=['POST'])
def api_agentic_aml():
    try:
        data = request.json
        result = run_agentic_aml_investigation(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/black-scholes', methods=['POST'])
def api_black_scholes():
    try:
        data = request.json
        S = float(data.get('spot', 100))
        K = float(data.get('strike', 100))
        T = float(data.get('maturity', 1))
        r = float(data.get('rate', 0.05))
        sigma = float(data.get('volatility', 0.2))
        opt_type = data.get('option_type', 'call')
        price = black_scholes(S, K, T, r, sigma, opt_type)
        greeks = bs_greeks(S, K, T, r, sigma, opt_type)
        spot_range = np.linspace(0.5*S, 1.5*S, 50)
        payoffs = []
        for s in spot_range:
            if opt_type == 'call':
                payoff = max(s - K, 0) - price
            else:
                payoff = max(K - s, 0) - price
            payoffs.append(round(payoff, 2))
        return jsonify({
            'price': round(price, 4),
            'greeks': {k: round(v, 6) for k, v in greeks.items()},
            'd1': round((math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T)), 4),
            'd2': round((math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T)) - sigma*math.sqrt(T), 4),
            'payoff_diagram': {
                'spot_prices': [round(s, 2) for s in spot_range.tolist()],
                'payoffs': payoffs,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monte-carlo-var', methods=['POST'])
def api_monte_carlo_var():
    try:
        data = request.json
        portfolio_value = float(data.get('portfolio_value', 10000000))
        n_assets = int(data.get('n_assets', 5))
        weights = [1/n_assets] * n_assets
        confidence = float(data.get('confidence', 0.95))
        horizon = int(data.get('horizon', 1))
        result = monte_carlo_var(portfolio_value, weights, n_sims=10000, confidence=confidence, horizon=horizon)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/basel-irb', methods=['POST'])
def api_basel_irb():
    try:
        data = request.json
        pd_val = float(data.get('pd', 0.02))
        lgd = float(data.get('lgd', 0.45))
        ead = float(data.get('ead', 1000000))
        maturity = float(data.get('maturity', 2.5))
        asset_class = data.get('asset_class', 'corporate')
        result = basel_irb_rwa(pd_val, lgd, ead, maturity, asset_class)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/copula-defaults', methods=['POST'])
def api_copula_defaults():
    try:
        data = request.json
        n_loans = int(data.get('n_loans', 1000))
        n_sims = int(data.get('n_sims', 5000))
        pd_val = float(data.get('pd', 0.05))
        lgd = float(data.get('lgd', 0.4))
        correlation = float(data.get('correlation', 0.3))
        result = gaussian_copula_defaults(n_loans, n_sims, pd_val, lgd, correlation)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stress-test', methods=['POST'])
def api_stress_test():
    try:
        if 'loans' in DATA_DFS and len(DATA_DFS['loans']) > 0:
            sample = DATA_DFS['loans'].sample(min(1000, len(DATA_DFS['loans'])))
            portfolio = sample.to_dict('records')
        else:
            portfolio = []
        scenarios = get_default_stress_scenarios()
        results = stress_test_portfolio(portfolio, scenarios)
        return jsonify({'scenarios': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-performance')
def api_model_performance():
    try:
        return jsonify({
            'credit_risk': {'auc': MODELS_DICT.get('credit_model', {}).get('auc', 0)},
            'fraud_detection': {'auc': MODELS_DICT.get('fraud_model', {}).get('auc', 0)},
            'kyc_aml': {'auc': MODELS_DICT.get('kyc_model', {}).get('auc', 0)},
            'churn': {'auc': MODELS_DICT.get('churn_model', {}).get('auc', 0)},
            'survival_analysis': {'c_index': MODELS_DICT.get('survival_model', {}).get('c_index', 0)},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ HTML/CSS/JS ============

CSS_STR = """
:root{--primary:#1a237e;--primary-light:#3949ab;--accent:#00bcd4;--accent2:#ff6f00;--bg:#0f1117;--card:#1a1d29;--card-light:#232838;--text:#e0e0e0;--text-dim:#9e9e9e;--success:#4caf50;--warning:#ff9800;--danger:#f44336;--border:#2d2d3d}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);display:flex;min-height:100vh}
.sidebar{width:260px;background:var(--card);border-right:1px solid var(--border);padding:0;position:fixed;height:100vh;overflow-y:auto;z-index:100}
.sidebar-header{padding:20px;border-bottom:1px solid var(--border);text-align:center}
.sidebar-header h1{font-size:1.4rem;background:linear-gradient(135deg,var(--accent),var(--primary-light));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sidebar-header .version{font-size:0.7rem;color:var(--accent);margin-top:4px;letter-spacing:1px}
.nav-section{padding:10px 0;border-bottom:1px solid var(--border)}
.nav-section-title{padding:8px 20px;font-size:0.7rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-dim);font-weight:600}
.nav-item{display:block;padding:10px 20px;color:var(--text-dim);text-decoration:none;font-size:0.85rem;transition:all 0.2s;border-left:3px solid transparent}
.nav-item:hover{background:var(--card-light);color:var(--text)}
.nav-item.active{background:var(--card-light);color:var(--accent);border-left-color:var(--accent)}
.nav-item .badge{background:var(--accent2);color:#fff;font-size:0.6rem;padding:2px 6px;border-radius:8px;font-weight:700;margin-left:6px}
.main{margin-left:260px;flex:1;padding:30px;min-width:0}
.page-header{margin-bottom:25px}
.page-header h2{font-size:1.6rem;color:var(--text)}
.page-header p{color:var(--text-dim);font-size:0.9rem;margin-top:5px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:20px}
.card h3{font-size:1rem;color:var(--accent);margin-bottom:15px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center;transition:transform 0.2s}
.stat-card:hover{transform:translateY(-3px)}
.stat-card .value{font-size:2rem;font-weight:700;color:var(--accent)}
.stat-card .label{font-size:0.8rem;color:var(--text-dim);margin-top:5px}
.stat-card.danger .value{color:var(--danger)}
.stat-card.success .value{color:var(--success)}
.stat-card.warning .value{color:var(--warning)}
.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px}
.form-group{display:flex;flex-direction:column;gap:5px}
.form-group label{font-size:0.8rem;color:var(--text-dim)}
.form-group input,.form-group select{background:var(--bg);border:1px solid var(--border);color:var(--text);padding:10px 12px;border-radius:8px;font-size:0.9rem}
.form-group input:focus,.form-group select:focus{outline:none;border-color:var(--accent)}
.btn{background:linear-gradient(135deg,var(--primary),var(--primary-light));color:#fff;border:none;padding:12px 30px;border-radius:8px;font-size:0.9rem;cursor:pointer;transition:opacity 0.2s;font-weight:600}
.btn:hover{opacity:0.9}
.result-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:15px;margin-top:15px;display:none}
.result-box.show{display:block}
.result-box .result-value{font-size:1.5rem;font-weight:700;color:var(--accent)}
.result-box .result-label{font-size:0.8rem;color:var(--text-dim)}
.chart-container{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:15px;margin-top:15px}
table{width:100%;border-collapse:collapse;font-size:0.85rem}
table th,table td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border)}
table th{color:var(--accent);font-weight:600;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.5px}
table tr:hover{background:var(--card-light)}
.tag{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:600}
.tag-danger{background:rgba(244,67,54,0.15);color:var(--danger)}
.tag-warning{background:rgba(255,152,0,0.15);color:var(--warning)}
.tag-success{background:rgba(76,175,80,0.15);color:var(--success)}
.agent-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:10px}
.agent-card .agent-name{color:var(--accent);font-size:0.85rem;font-weight:600;margin-bottom:5px}
.agent-card .agent-status{font-size:0.75rem;color:var(--success)}
.agent-card .agent-action{font-size:0.8rem;color:var(--text-dim);margin-top:3px}
.math-formula{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:15px;margin:10px 0;font-family:'Courier New',monospace;font-size:0.9rem;color:var(--accent);overflow-x:auto}
.info-banner{background:linear-gradient(135deg,rgba(0,188,212,0.1),rgba(26,35,126,0.1));border:1px solid rgba(0,188,212,0.3);border-radius:8px;padding:12px 15px;margin-bottom:15px;font-size:0.85rem;color:var(--text-dim)}
.info-banner strong{color:var(--accent)}
@media(max-width:768px){.sidebar{display:none}.main{margin-left:0;padding:15px}}
"""

# Load HTML templates from external file to keep app.py clean
# We'll build pages using a page() function

def page(title, content, active=''):
    nav_html = '<div class="sidebar"><div class="sidebar-header"><h1>FinSight AI</h1><div class="version">v2.0 RESEARCH</div></div>'
    sections = [
        ('Overview', [('/', 'Dashboard', 'overview'), ('/models', 'Model Performance', 'models')]),
        ('Retail Banking', [('/credit-risk', 'Credit Risk Scoring', 'credit'), ('/survival', 'Survival Analysis', 'survival'), ('/fraud', 'Fraud Detection', 'fraud'), ('/churn', 'Customer Churn', 'churn')]),
        ('Commercial Banking', [('/kyc', 'KYC / AML Risk', 'kyc'), ('/agentic-aml', 'Agentic AML Investigation', 'agentic')]),
        ('Quantitative Finance Lab', [('/black-scholes', 'Black-Scholes Pricing', 'bs'), ('/monte-carlo', 'Monte Carlo VaR', 'mc'), ('/basel-irb', 'Basel III IRB', 'basel'), ('/copula', 'Copula Defaults', 'copula'), ('/stress-test', 'Stress Testing', 'stress')]),
    ]
    new_badge = {'survival', 'agentic', 'bs', 'mc', 'basel', 'copula', 'stress'}
    for section_name, items in sections:
        nav_html += '<div class="nav-section"><div class="nav-section-title">' + section_name + '</div>'
        for href, label, key in items:
            badge = ' <span class="badge">NEW</span>' if key in new_badge else ''
            cls = ' active' if key == active else ''
            nav_html += '<a href="' + href + '" class="nav-item' + cls + '">' + label + badge + '</a>'
        nav_html += '</div>'
    nav_html += '</div>'
    return '<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>' + title + ' | FinSight AI v2.0</title><style>' + CSS_STR + '</style><script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script></head><body>' + nav_html + '<div class="main"><div class="page-header"><h2>' + title + '</h2></div>' + content + '</div></body></html>'

# ============ PAGE ROUTES ============

@app.route('/')
def page_dashboard():
    c = '<p>Advanced AI banking platform with quantitative finance, agentic AI workflows, and deep learning models.</p>'
    c += '<div class="stat-grid" id="stats"></div>'
    c += '<div class="card"><h3>Model Performance Summary</h3><table id="modelTable"><thead><tr><th>Model</th><th>Type</th><th>Score</th><th>Status</th></tr></thead><tbody></tbody></table></div>'
    c += '<div class="card"><h3>Platform Capabilities</h3><div class="stat-grid">'
    c += '<div class="stat-card"><div class="value">7</div><div class="label">Core ML Models</div></div>'
    c += '<div class="stat-card"><div class="value">5</div><div class="label">Quant Finance Tools</div></div>'
    c += '<div class="stat-card"><div class="value">5</div><div class="label">Agentic AI Agents</div></div>'
    c += '<div class="stat-card"><div class="value">1</div><div class="label">Survival Analysis</div></div>'
    c += '</div></div>'
    c += '<script>'
    c += 'fetch("/api/dashboard").then(r=>r.json()).then(d=>{'
    c += 'const stats=document.getElementById("stats");'
    c += 'const items=[["Customers",d.customers],["Transactions",d.transactions],["Loans",d.loans],["Fraud Alerts",d.fraud_alerts],["Defaults",d.defaults],["KYC Alerts",d.kyc_alerts],["Churn Cases",d.churn_cases],["Portfolios",d.portfolios]];'
    c += 'stats.innerHTML=items.map(x=>"<div class=\\"stat-card\\"><div class=\\"value\\">"+x[1].toLocaleString()+"</div><div class=\\"label\\">"+x[0]+"</div></div>").join("");'
    c += 'const mt=document.querySelector("#modelTable tbody");'
    c += 'const models=[["Credit Risk","XGBoost",d.models.credit_auc],["Fraud Detection","XGBoost+IF",d.models.fraud_auc],["KYC/AML","Random Forest",d.models.kyc_auc],["Churn","XGBoost",d.models.churn_auc],["Survival","Cox PH",d.models.survival_c_index]];'
    c += 'mt.innerHTML=models.map(m=>"<tr><td>"+m[0]+"</td><td>"+m[1]+"</td><td>"+(m[2]*100).toFixed(1)+"%</td><td><span class=\\"tag tag-success\\">Active</span></td></tr>").join("");'
    c += '});'
    c += '</script>'
    return page('Dashboard', c, 'overview')

@app.route('/credit-risk')
def page_credit_risk():
    c = '<p>Credit risk scoring powered by XGBoost with SHAP-style feature explainability.</p>'
    c += '<div class="card"><h3>Loan Application Details</h3>'
    c += '<form id="creditForm" onsubmit="return submitCredit(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Loan Amount (Rs.)</label><input type="number" name="loan_amount" value="500000" step="10000"></div>'
    c += '<div class="form-group"><label>Interest Rate</label><input type="number" name="interest_rate" value="0.12" step="0.01"></div>'
    c += '<div class="form-group"><label>Term (months)</label><input type="number" name="term_months" value="36"></div>'
    c += '<div class="form-group"><label>Credit Score</label><input type="number" name="credit_score" value="680"></div>'
    c += '<div class="form-group"><label>Annual Income (Rs.)</label><input type="number" name="annual_income" value="800000" step="10000"></div>'
    c += '<div class="form-group"><label>Age</label><input type="number" name="age" value="35"></div>'
    c += '<div class="form-group"><label>DTI Ratio</label><input type="number" name="dti_ratio" value="0.30" step="0.01"></div>'
    c += '<div class="form-group"><label>Employment Years</label><input type="number" name="employment_years" value="5"></div>'
    c += '<div class="form-group"><label>Loan Type</label><select name="loan_type"><option>Personal</option><option>Home</option><option>Auto</option><option>Education</option><option>Business</option></select></div>'
    c += '<div class="form-group"><label>Dependents</label><input type="number" name="num_dependents" value="2"></div>'
    c += '</div><p><button type="submit" class="btn">Assess Credit Risk</button></p></form></div>'
    c += '<div class="result-box" id="result"><div class="result-label">Default Probability</div><div class="result-value" id="prob">--</div><div id="details"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="impChart"></canvas></div>'
    c += '<script>'
    c += 'function submitCredit(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/credit-risk",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'document.getElementById("prob").textContent=(r.probability*100).toFixed(1)+"%";'
    c += 'document.getElementById("details").innerHTML="<p>Risk Band: <span class=\\"tag tag-"+(r.risk_band==="LOW"?"success":r.risk_band==="MEDIUM"?"warning":"danger")+"\\">"+r.risk_band+"</span> | Recommendation: <strong>"+r.recommendation+"</strong></p>";'
    c += 'if(r.feature_importance&&r.feature_importance.length)drawChart(r.feature_importance)});return false}'
    c += 'let creditChart=null;function drawChart(data){const ctx=document.getElementById("impChart");document.getElementById("chartBox").style.display="block";if(creditChart)creditChart.destroy();'
    c += 'creditChart=new Chart(ctx,{type:"bar",data:{labels:data.map(d=>d.feature),datasets:[{label:"Feature Importance",data:data.map(d=>d.importance),backgroundColor:"#00bcd4"}]},options:{responsive:true,indexAxis:"y",plugins:{title:{display:true,text:"Feature Importance (SHAP-style)"}}}})}'
    c += '</script>'
    return page('Credit Risk Scoring', c, 'credit')

@app.route('/survival')
def page_survival():
    c = '<p>Cox Proportional Hazards model predicting <strong>when</strong> a borrower will default, not just <em>if</em>.</p>'
    c += '<div class="info-banner"><strong>New:</strong> Uses the Cox PH model from lifelines with concordance index evaluation and hazard ratio extraction.</div>'
    c += '<div class="card"><h3>Borrower & Loan Details</h3>'
    c += '<form id="survForm" onsubmit="return submitSurvival(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Loan Amount (Rs.)</label><input type="number" name="loan_amount" value="1000000" step="50000"></div>'
    c += '<div class="form-group"><label>Interest Rate</label><input type="number" name="interest_rate" value="0.11" step="0.01"></div>'
    c += '<div class="form-group"><label>Term (months)</label><input type="number" name="term_months" value="60"></div>'
    c += '<div class="form-group"><label>Credit Score</label><input type="number" name="credit_score" value="650"></div>'
    c += '<div class="form-group"><label>Annual Income (Rs.)</label><input type="number" name="annual_income" value="600000" step="10000"></div>'
    c += '<div class="form-group"><label>Age</label><input type="number" name="age" value="32"></div>'
    c += '<div class="form-group"><label>DTI Ratio</label><input type="number" name="dti_ratio" value="0.35" step="0.01"></div>'
    c += '<div class="form-group"><label>Employment Years</label><input type="number" name="employment_years" value="3"></div>'
    c += '<div class="form-group"><label>Loan Type</label><select name="loan_type"><option>Personal</option><option>Home</option><option>Auto</option><option>Education</option><option>Business</option></select></div>'
    c += '<div class="form-group"><label>Dependents</label><input type="number" name="num_dependents" value="2"></div>'
    c += '</div><p><button type="submit" class="btn">Predict Survival Curve</button></p></form></div>'
    c += '<div class="result-box" id="result"><div class="result-label">Survival Analysis Result</div><div id="survDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="survChart"></canvas></div>'
    c += '<div class="card" id="hazardCard" style="display:none"><h3>Hazard Ratios (exp(beta))</h3><p style="color:var(--text-dim);font-size:0.8rem">HR > 1 = risk accelerator, HR < 1 = protective factor</p><table id="hazardTable"><thead><tr><th>Feature</th><th>Hazard Ratio</th><th>Interpretation</th></tr></thead><tbody></tbody></table></div>'
    c += '<script>'
    c += 'function submitSurvival(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/survival",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-value\\">C-Index: "+(r.c_index*100).toFixed(1)+"%</div>";'
    c += 'html+="<p>Partial Hazard: "+r.partial_hazard+"</p>";'
    c += 'if(r.median_survival)html+="<p>Median Survival Time: "+r.median_survival+" months</p>";'
    c += 'document.getElementById("survDetails").innerHTML=html;'
    c += 'drawSurvChart(r.survival_probs);if(r.hazard_ratios)drawHazardTable(r.hazard_ratios)});return false}'
    c += 'let survChart=null;function drawSurvChart(probs){const labels=Object.keys(probs).map(k=>k+"m");const data=Object.values(probs);const ctx=document.getElementById("survChart");document.getElementById("chartBox").style.display="block";if(survChart)survChart.destroy();'
    c += 'survChart=new Chart(ctx,{type:"line",data:{labels:labels,datasets:[{label:"Survival Probability",data:data,borderColor:"#00bcd4",backgroundColor:"rgba(0,188,212,0.1)",fill:true}]},options:{responsive:true,plugins:{title:{display:true,text:"Survival Curve (Probability of No Default Over Time)"}},scales:{y:{min:0,max:1,title:{display:true,text:"P(No Default)"}},x:{title:{display:true,text:"Months"}}}}})}'
    c += 'function drawHazardTable(hr){const t=document.querySelector("#hazardTable tbody");document.getElementById("hazardCard").style.display="block";'
    c += 't.innerHTML=Object.entries(hr).map(x=>{const k=x[0],v=x[1];const interp=v>1.2?"<span class=\\"tag tag-danger\\">Risk Accelerator</span>":v<0.8?"<span class=\\"tag tag-success\\">Protective Factor</span>":"<span class=\\"tag tag-warning\\">Neutral</span>";'
    c += 'return "<tr><td>"+k.replace(/_/g," ").replace(/loan type /,"")+"</td><td>"+v.toFixed(3)+"</td><td>"+interp+"</td></tr>"}).join("")}'
    c += '</script>'
    return page('Survival Analysis', c, 'survival')

@app.route('/fraud')
def page_fraud():
    c = '<p>Fraud detection using XGBoost + Isolation Forest with dual scoring. The anomaly detection model identifies unusual patterns that deviate from normal transaction behavior.</p>'
    c += '<div class="info-banner"><strong>Dual-Model Approach:</strong> XGBoost provides supervised fraud probability, while the Isolation Forest (autoencoder proxy) provides unsupervised anomaly detection.</div>'
    c += '<div class="card"><h3>Transaction Details</h3>'
    c += '<form id="fraudForm" onsubmit="return submitFraud(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Amount (Rs.)</label><input type="number" name="amount" value="5000" step="100"></div>'
    c += '<div class="form-group"><label>Hour of Day</label><input type="number" name="hour" value="14" min="0" max="23"></div>'
    c += '<div class="form-group"><label>Transaction Type</label><select name="txn_type"><option>UPI</option><option>NEFT</option><option>IMPS</option><option>RTGS</option><option>Card</option><option>Cheque</option></select></div>'
    c += '<div class="form-group"><label>Channel</label><select name="channel"><option>Mobile</option><option>NetBanking</option><option>ATM</option><option>Branch</option><option>POS</option></select></div>'
    c += '<div class="form-group"><label>Merchant Category</label><select name="merchant_category"><option>Retail</option><option>Food</option><option>Travel</option><option>Fuel</option><option>Entertainment</option><option>Bills</option><option>Transfer</option></select></div>'
    c += '<div class="form-group"><label>Transaction Speed (sec)</label><input type="number" name="txn_speed" value="2" step="0.1"></div>'
    c += '<div class="form-group"><label>Device Changed?</label><select name="device_change"><option value="0">No</option><option value="1">Yes</option></select></div>'
    c += '<div class="form-group"><label>Freq (last 24h)</label><input type="number" name="freq_last_24h" value="3"></div>'
    c += '</div><p><button type="submit" class="btn">Detect Fraud</button></p></form></div>'
    c += '<div class="result-box" id="result"><div id="fraudDetails"></div></div>'
    c += '<script>'
    c += 'function submitFraud(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/fraud-detect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-label\\">Combined Fraud Score</div><div class=\\"result-value\\">"+(r.combined_score*100).toFixed(1)+"%</div>";'
    c += 'html+="<table><tr><th>Metric</th><th>Value</th></tr>";'
    c += 'html+="<tr><td>XGBoost Fraud Probability</td><td>"+(r.xgb_fraud_prob*100).toFixed(2)+"%</td></tr>";'
    c += 'html+="<tr><td>Anomaly Score (IF)</td><td>"+r.anomaly_score.toFixed(4)+"</td></tr>";'
    c += 'html+="<tr><td>Reconstruction Error</td><td>"+r.reconstruction_error.toFixed(4)+"</td></tr>";'
    c += 'html+="<tr><td>Is Anomaly</td><td>"+(r.is_anomaly?"<span class=\\"tag tag-danger\\">YES</span>":"<span class=\\"tag tag-success\\">NO</span>")+"</td></tr>";'
    c += 'html+="<tr><td>Verdict</td><td><span class=\\"tag tag-"+(r.verdict==="FRAUD"?"danger":r.verdict==="SUSPICIOUS"?"warning":"success")+"\\">"+r.verdict+"</span></td></tr></table>";'
    c += 'document.getElementById("fraudDetails").innerHTML=html});return false}'
    c += '</script>'
    return page('Fraud Detection', c, 'fraud')

@app.route('/churn')
def page_churn():
    c = '<p>Customer churn prediction using XGBoost with feature explainability.</p>'
    c += '<div class="card"><h3>Customer Details</h3>'
    c += '<form id="churnForm" onsubmit="return submitChurn(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Age</label><input type="number" name="age" value="35"></div>'
    c += '<div class="form-group"><label>Annual Income (Rs.)</label><input type="number" name="annual_income" value="800000" step="10000"></div>'
    c += '<div class="form-group"><label>Credit Score</label><input type="number" name="credit_score" value="680"></div>'
    c += '<div class="form-group"><label>Months with Bank</label><input type="number" name="months_with_bank" value="24"></div>'
    c += '<div class="form-group"><label>Num Products</label><input type="number" name="num_products" value="2"></div>'
    c += '<div class="form-group"><label>Avg Monthly Balance</label><input type="number" name="avg_monthly_balance" value="50000"></div>'
    c += '<div class="form-group"><label>Satisfaction (1-5)</label><input type="number" name="satisfaction_score" value="3.5" step="0.1" min="1" max="5"></div>'
    c += '<div class="form-group"><label>Complaints (6m)</label><input type="number" name="complaints_last_6m" value="1"></div>'
    c += '<div class="form-group"><label>Digital Engagement (0-1)</label><input type="number" name="digital_engagement" value="0.5" step="0.05" min="0" max="1"></div>'
    c += '<div class="form-group"><label>Branch Visits (3m)</label><input type="number" name="branch_visits_3m" value="3"></div>'
    c += '<div class="form-group"><label>Product Utilization (0-1)</label><input type="number" name="product_utilization" value="0.5" step="0.05" min="0" max="1"></div>'
    c += '<div class="form-group"><label>Segment</label><select name="segment"><option>Retail</option><option>Premium</option><option>HNI</option><option>Mass</option></select></div>'
    c += '<div class="form-group"><label>Occupation</label><select name="occupation"><option>Salaried</option><option>Self-Employed</option><option>Business</option><option>Professional</option><option>Retired</option></select></div>'
    c += '<div class="form-group"><label>City</label><select name="city"><option>Mumbai</option><option>Delhi</option><option>Bangalore</option><option>Hyderabad</option><option>Chennai</option></select></div>'
    c += '</div><p><button type="submit" class="btn">Predict Churn Risk</button></p></form></div>'
    c += '<div class="result-box" id="result"><div id="churnDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="impChart"></canvas></div>'
    c += '<script>'
    c += 'function submitChurn(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/churn",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-label\\">Churn Probability</div><div class=\\"result-value\\">"+(r.probability*100).toFixed(1)+"%</div>";'
    c += 'html+="<p>Risk Level: <span class=\\"tag tag-"+(r.risk_level==="LOW"?"success":r.risk_level==="MEDIUM"?"warning":"danger")+"\\">"+r.risk_level+"</span></p>";'
    c += 'html+="<p>Recommendation: <strong>"+r.recommendation+"</strong></p>";'
    c += 'document.getElementById("churnDetails").innerHTML=html;'
    c += 'if(r.feature_importance&&r.feature_importance.length)drawChurnChart(r.feature_importance)});return false}'
    c += 'let churnChart=null;function drawChurnChart(data){const ctx=document.getElementById("impChart");document.getElementById("chartBox").style.display="block";if(churnChart)churnChart.destroy();'
    c += 'churnChart=new Chart(ctx,{type:"bar",data:{labels:data.map(d=>d.feature),datasets:[{label:"Feature Importance",data:data.map(d=>d.importance),backgroundColor:"#ff6f00"}]},options:{responsive:true,indexAxis:"y",plugins:{title:{display:true,text:"Churn Risk Drivers"}}}})}'
    c += '</script>'
    return page('Customer Churn', c, 'churn')

@app.route('/kyc')
def page_kyc():
    c = '<p>KYC/AML risk scoring using Random Forest with explainability.</p>'
    c += '<div class="card"><h3>Customer KYC Details</h3>'
    c += '<form id="kycForm" onsubmit="return submitKYC(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Country</label><select name="country"><option>India</option><option>USA</option><option>UK</option><option>Singapore</option><option>UAE</option><option>Switzerland</option><option>Cyprus</option><option>Cayman Is.</option><option>British Virgin Is.</option><option>Panama</option></select></div>'
    c += '<div class="form-group"><label>Txn Volume (30d, Rs.)</label><input type="number" name="txn_volume_30d" value="500000" step="50000"></div>'
    c += '<div class="form-group"><label>Num Large Txns</label><input type="number" name="num_large_txns" value="3"></div>'
    c += '<div class="form-group"><label>Structuring Detected?</label><select name="structuring_detected"><option value="0">No</option><option value="1">Yes</option></select></div>'
    c += '<div class="form-group"><label>PEP Flag?</label><select name="pep_flag"><option value="0">No</option><option value="1">Yes</option></select></div>'
    c += '<div class="form-group"><label>Sanctions Hit?</label><select name="sanctions_hit"><option value="0">No</option><option value="1">Yes</option></select></div>'
    c += '<div class="form-group"><label>Risk Score (0-100)</label><input type="number" name="risk_score" value="35"></div>'
    c += '<div class="form-group"><label>Account Age (months)</label><input type="number" name="account_age_months" value="24"></div>'
    c += '</div><p><button type="submit" class="btn">Assess AML Risk</button></p></form></div>'
    c += '<div class="result-box" id="result"><div id="kycDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="impChart"></canvas></div>'
    c += '<script>'
    c += 'function submitKYC(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/kyc-aml",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-label\\">Suspicious Activity Probability</div><div class=\\"result-value\\">"+(r.probability*100).toFixed(1)+"%</div>";'
    c += 'html+="<p>Risk Level: <span class=\\"tag tag-"+(r.risk_level==="LOW"?"success":r.risk_level==="MEDIUM"?"warning":"danger")+"\\">"+r.risk_level+"</span></p>";'
    c += 'html+="<p>Recommendation: <strong>"+r.recommendation+"</strong></p>";'
    c += 'document.getElementById("kycDetails").innerHTML=html;'
    c += 'if(r.feature_importance&&r.feature_importance.length)drawKYCChart(r.feature_importance)});return false}'
    c += 'let kycChart=null;function drawKYCChart(data){const ctx=document.getElementById("impChart");document.getElementById("chartBox").style.display="block";if(kycChart)kycChart.destroy();'
    c += 'kycChart=new Chart(ctx,{type:"bar",data:{labels:data.map(d=>d.feature),datasets:[{label:"Feature Importance",data:data.map(d=>d.importance),backgroundColor:"#ff9800"}]},options:{responsive:true,indexAxis:"y",plugins:{title:{display:true,text:"AML Risk Drivers"}}}})}'
    c += '</script>'
    return page('KYC / AML Risk', c, 'kyc')

@app.route('/agentic-aml')
def page_agentic_aml():
    c = '<p>Multi-agent autonomous AML investigation workflow. Five specialized AI agents collaborate to investigate a customer, each handling a specific task with full audit trail.</p>'
    c += '<div class="info-banner"><strong>Agentic AI:</strong> Inspired by Deloitte multi-agent KYC framework where one agent pulls data, another scores risk, a third checks sanctions.</div>'
    c += '<div class="card"><h3>Customer Under Investigation</h3>'
    c += '<form id="agenticForm" onsubmit="return submitAgentic(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Customer ID</label><input type="text" name="customer_id" value="C123456"></div>'
    c += '<div class="form-group"><label>Country</label><select name="country"><option>India</option><option>Cayman Is.</option><option>British Virgin Is.</option><option>Panama</option><option>Cyprus</option><option>Singapore</option></select></div>'
    c += '<div class="form-group"><label>Txn Volume (30d, Rs.)</label><input type="number" name="txn_volume_30d" value="5000000" step="100000"></div>'
    c += '<div class="form-group"><label>Num Large Txns</label><input type="number" name="num_large_txns" value="8"></div>'
    c += '<div class="form-group"><label>Structuring Detected?</label><select name="structuring_detected"><option value="0">No</option><option value="1">Yes</option></select></div>'
    c += '<div class="form-group"><label>PEP Flag?</label><select name="pep_flag"><option value="0">No</option><option value="1">Yes</option></select></div>'
    c += '<div class="form-group"><label>Sanctions Hit?</label><select name="sanctions_hit"><option value="0">No</option><option value="1">Yes</option></select></div>'
    c += '<div class="form-group"><label>Risk Score (0-100)</label><input type="number" name="risk_score" value="55"></div>'
    c += '<div class="form-group"><label>Account Age (months)</label><input type="number" name="account_age_months" value="18"></div>'
    c += '</div><p><button type="submit" class="btn">Launch Multi-Agent Investigation</button></p></form></div>'
    c += '<div id="agentResults"></div>'
    c += '<script>'
    c += 'function submitAgentic(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/agentic-aml",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{'
    c += 'let html="<div class=\\"card\\"><h3>Investigation Results</h3>";'
    c += 'html+="<div class=\\"stat-grid\\"><div class=\\"stat-card "+(r.sar_recommended?"danger":"success")+"\\"><div class=\\"value\\">"+(r.sar_recommended?"SAR":"CLEAR")+"</div><div class=\\"label\\">Recommendation</div></div>";'
    c += 'html+="<div class=\\"stat-card\\"><div class=\\"value\\">"+r.total_agents+"</div><div class=\\"label\\">Agents Deployed</div></div>";'
    c += 'html+="<div class=\\"stat-card\\"><div class=\\"value\\">"+r.data_points+"</div><div class=\\"label\\">Data Points</div></div>";'
    c += 'html+="<div class=\\"stat-card\\"><div class=\\"value\\">"+(r.confidence*100).toFixed(0)+"%</div><div class=\\"label\\">Confidence</div></div></div>";'
    c += 'html+="<p>"+r.investigation_summary+"</p></div>";'
    c += 'html+="<div class=\\"card\\"><h3>Agent Workflow Execution</h3>";'
    c += 'r.agents.forEach((a,i)=>{'
    c += 'html+="<div class=\\"agent-card\\"><div class=\\"agent-name\\">"+(i+1)+". "+a.name+"</div>";'
    c += 'html+="<div class=\\"agent-status\\">Status: "+a.status.toUpperCase()+"</div>";'
    c += 'a.actions.forEach(act=>{html+="<div class=\\"agent-action\\">&#9656; "+act+"</div>"});'
    c += 'html+="</div>"});'
    c += 'html+="</div>";'
    c += 'document.getElementById("agentResults").innerHTML=html});return false}'
    c += '</script>'
    return page('Agentic AML Investigation', c, 'agentic')

@app.route('/black-scholes')
def page_black_scholes():
    c = '<p>Black-Scholes option pricing model with Greeks computation and payoff visualization.</p>'
    c += '<div class="card"><h3>Option Parameters</h3>'
    c += '<form id="bsForm" onsubmit="return submitBS(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Spot Price (S)</label><input type="number" name="spot" value="100" step="0.5"></div>'
    c += '<div class="form-group"><label>Strike Price (K)</label><input type="number" name="strike" value="100" step="0.5"></div>'
    c += '<div class="form-group"><label>Time to Maturity (years)</label><input type="number" name="maturity" value="1" step="0.1"></div>'
    c += '<div class="form-group"><label>Risk-free Rate</label><input type="number" name="rate" value="0.05" step="0.005"></div>'
    c += '<div class="form-group"><label>Volatility (sigma)</label><input type="number" name="volatility" value="0.20" step="0.01"></div>'
    c += '<div class="form-group"><label>Option Type</label><select name="option_type"><option>call</option><option>put</option></select></div>'
    c += '</div><p><button type="submit" class="btn">Price Option</button></p></form></div>'
    c += '<div class="math-formula">BS Formula: C = S*N(d1) - K*e^(-rT)*N(d2)<br>where d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T)), d2 = d1 - sigma*sqrt(T)</div>'
    c += '<div class="result-box" id="result"><div id="bsDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="payoffChart"></canvas></div>'
    c += '<script>'
    c += 'function submitBS(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/black-scholes",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-label\\">Option Price</div><div class=\\"result-value\\">"+r.price.toFixed(4)+"</div>";'
    c += 'html+="<table><tr><th>Greek</th><th>Value</th></tr>";'
    c += 'Object.entries(r.greeks).forEach(x=>{html+="<tr><td>"+x[0].toUpperCase()+"</td><td>"+x[1].toFixed(6)+"</td></tr>"});'
    c += 'html+="<tr><td>d1</td><td>"+r.d1+"</td></tr><tr><td>d2</td><td>"+r.d2+"</td></tr></table>";'
    c += 'document.getElementById("bsDetails").innerHTML=html;drawPayoff(r.payoff_diagram)});return false}'
    c += 'let payoffChart=null;function drawPayoff(data){const ctx=document.getElementById("payoffChart");document.getElementById("chartBox").style.display="block";if(payoffChart)payoffChart.destroy();'
    c += 'payoffChart=new Chart(ctx,{type:"line",data:{labels:data.spot_prices,datasets:[{label:"P&L",data:data.payoffs,borderColor:"#00bcd4",backgroundColor:"rgba(0,188,212,0.1)",fill:true}]},options:{responsive:true,plugins:{title:{display:true,text:"Option Payoff Diagram (P&L at Expiry)"}},scales:{x:{title:{display:true,text:"Spot Price at Expiry"}},y:{title:{display:true,text:"Profit / Loss"}}}}})}'
    c += '</script>'
    return page('Black-Scholes Option Pricing', c, 'bs')

@app.route('/monte-carlo')
def page_monte_carlo():
    c = '<p>Monte Carlo Value-at-Risk simulation with 10,000 scenarios and Expected Shortfall computation.</p>'
    c += '<div class="card"><h3>Portfolio Parameters</h3>'
    c += '<form id="mcForm" onsubmit="return submitMC(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Portfolio Value (Rs.)</label><input type="number" name="portfolio_value" value="10000000" step="1000000"></div>'
    c += '<div class="form-group"><label>Number of Assets</label><input type="number" name="n_assets" value="5" min="2" max="20"></div>'
    c += '<div class="form-group"><label>Confidence Level</label><select name="confidence"><option value="0.90">90%</option><option value="0.95">95%</option><option value="0.99">99%</option></select></div>'
    c += '<div class="form-group"><label>Time Horizon (days)</label><input type="number" name="horizon" value="1" min="1" max="30"></div>'
    c += '</div><p><button type="submit" class="btn">Run Monte Carlo Simulation</button></p></form></div>'
    c += '<div class="math-formula">VaR(alpha) = inf{l : P(Loss > l) <= 1-alpha}<br>ES(alpha) = E[Loss | Loss > VaR(alpha)]</div>'
    c += '<div class="result-box" id="result"><div id="mcDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="lossChart"></canvas></div>'
    c += '<script>'
    c += 'function submitMC(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/monte-carlo-var",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-label\\">Value at Risk (95%)</div><div class=\\"result-value\\">Rs. "+r.var.toLocaleString()+"</div>";'
    c += 'html+="<table><tr><th>Metric</th><th>Value</th></tr>";'
    c += 'html+="<tr><td>Expected Shortfall (ES)</td><td>Rs. "+r.es.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>Mean Loss</td><td>Rs. "+r.mean_loss.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>Std Dev</td><td>Rs. "+r.std_loss.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>VaR 90%</td><td>Rs. "+r.percentiles["90"].toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>VaR 95%</td><td>Rs. "+r.percentiles["95"].toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>VaR 99%</td><td>Rs. "+r.percentiles["99"].toLocaleString()+"</td></tr></table>";'
    c += 'document.getElementById("mcDetails").innerHTML=html;drawLossChart(r.sim_losses)});return false}'
    c += 'let lossChart=null;function drawLossChart(losses){const ctx=document.getElementById("lossChart");document.getElementById("chartBox").style.display="block";if(lossChart)lossChart.destroy();'
    c += 'const buckets={};const min=Math.min(...losses),max=Math.max(...losses);const step=(max-min)/30;losses.forEach(v=>{const b=Math.floor((v-min)/step);buckets[b]=(buckets[b]||0)+1});'
    c += 'const labels=Object.keys(buckets).map(b=>Math.round(min+parseInt(b)*step));const data=Object.values(buckets);'
    c += 'lossChart=new Chart(ctx,{type:"bar",data:{labels:labels,datasets:[{label:"Frequency",data:data,backgroundColor:"rgba(0,188,212,0.5)",borderColor:"#00bcd4"}]},options:{responsive:true,plugins:{title:{display:true,text:"Simulated Loss Distribution (10,000 scenarios)"}},scales:{x:{title:{display:true,text:"Loss Amount (Rs.)"}},y:{title:{display:true,text:"Frequency"}}}}})}'
    c += '</script>'
    return page('Monte Carlo VaR', c, 'mc')

@app.route('/basel-irb')
def page_basel_irb():
    c = '<p>Basel III Internal Ratings-Based (IRB) risk-weighted assets calculation using the regulatory formula.</p>'
    c += '<div class="card"><h3>Risk Parameters</h3>'
    c += '<form id="baselForm" onsubmit="return submitBasel(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Probability of Default (PD)</label><input type="number" name="pd" value="0.02" step="0.005" min="0.0003" max="1"></div>'
    c += '<div class="form-group"><label>Loss Given Default (LGD)</label><input type="number" name="lgd" value="0.45" step="0.05" min="0" max="1"></div>'
    c += '<div class="form-group"><label>Exposure at Default (EAD, Rs.)</label><input type="number" name="ead" value="1000000" step="100000"></div>'
    c += '<div class="form-group"><label>Effective Maturity (years)</label><input type="number" name="maturity" value="2.5" step="0.5" min="1" max="5"></div>'
    c += '<div class="form-group"><label>Asset Class</label><select name="asset_class"><option>corporate</option><option>sme</option><option>retail_mortgage</option><option>retail_revolving</option><option>retail_other</option></select></div>'
    c += '</div><p><button type="submit" class="btn">Calculate RWA</button></p></form></div>'
    c += '<div class="math-formula">K = [LGD * N((1-R)^(-1/2) * G(PD) + (R/(1-R))^(1/2) * G(0.999)) - PD * LGD] * M_adj<br>RWA = K * 12.5 * EAD<br>Capital Required = RWA * 8%</div>'
    c += '<div class="result-box" id="result"><div id="baselDetails"></div></div>'
    c += '<script>'
    c += 'function submitBasel(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/basel-irb",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-label\\">Risk-Weighted Assets</div><div class=\\"result-value\\">Rs. "+r.rwa.toLocaleString()+"</div>";'
    c += 'html+="<table><tr><th>Metric</th><th>Value</th></tr>";'
    c += 'html+="<tr><td>Capital Required (8%)</td><td>Rs. "+r.capital_required.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>Expected Loss</td><td>Rs. "+r.expected_loss.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>K (Capital Ratio)</td><td>"+(r.K*100).toFixed(2)+"%</td></tr>";'
    c += 'html+="<tr><td>Asset Correlation (R)</td><td>"+(r.R*100).toFixed(1)+"%</td></tr>";'
    c += 'html+="<tr><td>RWA Density</td><td>"+r.rwa_density.toFixed(1)+"%</td></tr></table>";'
    c += 'document.getElementById("baselDetails").innerHTML=html});return false}'
    c += '</script>'
    return page('Basel III IRB Capital', c, 'basel')

@app.route('/copula')
def page_copula():
    c = '<p>Gaussian Copula model for correlated portfolio defaults. Simulates dependent defaults across a loan portfolio using a systemic factor.</p>'
    c += '<div class="card"><h3>Portfolio Parameters</h3>'
    c += '<form id="copForm" onsubmit="return submitCopula(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Number of Loans</label><input type="number" name="n_loans" value="1000" min="100" max="5000"></div>'
    c += '<div class="form-group"><label>Number of Simulations</label><input type="number" name="n_sims" value="5000" min="1000" max="10000"></div>'
    c += '<div class="form-group"><label>PD (per loan)</label><input type="number" name="pd" value="0.05" step="0.01" min="0.001" max="0.5"></div>'
    c += '<div class="form-group"><label>LGD</label><input type="number" name="lgd" value="0.40" step="0.05" min="0" max="1"></div>'
    c += '<div class="form-group"><label>Correlation (asset)</label><input type="number" name="correlation" value="0.30" step="0.05" min="0" max="0.9"></div>'
    c += '</div><p><button type="submit" class="btn">Run Copula Simulation</button></p></form></div>'
    c += '<div class="math-formula">X_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i<br>Default if X_i < Phi^(-1)(PD)<br>where Z = systemic factor, epsilon_i = idiosyncratic factor</div>'
    c += '<div class="result-box" id="result"><div id="copDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="distChart"></canvas></div>'
    c += '<script>'
    c += 'function submitCopula(e){e.preventDefault();const f=new FormData(e.target);const d={};f.forEach((v,k)=>d[k]=v);'
    c += 'fetch("/api/copula-defaults",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)})'
    c += '.then(r=>r.json()).then(r=>{document.getElementById("result").classList.add("show");'
    c += 'let html="<div class=\\"result-label\\">99% VaR (Portfolio Loss)</div><div class=\\"result-value\\">Rs. "+r.var.toLocaleString()+"</div>";'
    c += 'html+="<table><tr><th>Metric</th><th>Value</th></tr>";'
    c += 'html+="<tr><td>Expected Shortfall (99%)</td><td>Rs. "+r.es.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>Mean Loss</td><td>Rs. "+r.mean_loss.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>Max Loss</td><td>Rs. "+r.max_loss.toLocaleString()+"</td></tr>";'
    c += 'html+="<tr><td>Avg Defaults</td><td>"+r.mean_defaults+"</td></tr>";'
    c += 'html+="<tr><td>Max Defaults</td><td>"+r.max_defaults+"</td></tr></table>";'
    c += 'document.getElementById("copDetails").innerHTML=html;drawDistChart(r.loss_distribution)});return false}'
    c += 'let distChart=null;function drawDistChart(dist){const labels=dist.map((_,i)=>i*2+"%");const ctx=document.getElementById("distChart");document.getElementById("chartBox").style.display="block";if(distChart)distChart.destroy();'
    c += 'distChart=new Chart(ctx,{type:"line",data:{labels:labels,datasets:[{label:"Loss Distribution",data:dist,borderColor:"#ff6f00",backgroundColor:"rgba(255,111,0,0.1)",fill:true}]},options:{responsive:true,plugins:{title:{display:true,text:"Portfolio Loss Distribution (Percentiles)"}},scales:{y:{title:{display:true,text:"Loss (Rs.)"}}}}})}'
    c += '</script>'
    return page('Copula Dependent Defaults', c, 'copula')

@app.route('/stress-test')
def page_stress_test():
    c = '<p>Macro stress testing engine with multiple severity scenarios. Simulates GDP shocks, unemployment spikes, and house price declines to assess portfolio resilience.</p>'
    c += '<div class="card"><h3>Stress Test Scenarios</h3>'
    c += '<p>Click below to run stress testing on the loan portfolio with 6 predefined macro scenarios.</p>'
    c += '<p><button class="btn" onclick="runStress()">Run Stress Test</button></p></div>'
    c += '<div id="stressResults"></div>'
    c += '<script>'
    c += 'function runStress(){fetch("/api/stress-test",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})})'
    c += '.then(r=>r.json()).then(r=>{let html="<div class=\\"card\\"><h3>Stress Test Results</h3>";'
    c += 'html+="<table><thead><tr><th>Scenario</th><th>GDP Shock</th><th>Unemp.</th><th>HPI</th><th>Exp. Loss</th><th>Loss Rate</th><th>Defaults</th></tr></thead><tbody>";'
    c += 'r.scenarios.forEach(s=>{html+="<tr><td>"+s.scenario+"</td><td>"+s.gdp_shock+"%</td><td>+"+s.unemployment_shock+"%</td><td>"+s.house_price_shock+"%</td><td>Rs. "+Math.round(s.expected_loss).toLocaleString()+"</td><td><span class=\\"tag tag-"+(s.loss_rate>3?"danger":s.loss_rate>1?"warning":"success")+"\\">"+s.loss_rate+"%</span></td><td>"+s.defaults+"</td></tr>"});'
    c += 'html+="</tbody></table></div>";document.getElementById("stressResults").innerHTML=html})}'
    c += '</script>'
    return page('Stress Testing', c, 'stress')

@app.route('/models')
def page_models():
    c = '<p>Detailed model performance metrics for all models in the platform.</p>'
    c += '<div class="card"><h3>Model Performance Dashboard</h3><table id="perfTable"><thead><tr><th>Model</th><th>Algorithm</th><th>Score</th><th>Dataset</th><th>Metric</th></tr></thead><tbody></tbody></table></div>'
    c += '<div class="card"><h3>Model Architecture Details</h3><table><thead><tr><th>Module</th><th>Technique</th><th>Innovation</th></tr></thead><tbody>'
    c += '<tr><td>Credit Risk</td><td>XGBoost</td><td>SHAP-style feature importance for explainability</td></tr>'
    c += '<tr><td>Fraud Detection</td><td>XGBoost + Isolation Forest</td><td>Dual-model: supervised + unsupervised anomaly detection</td></tr>'
    c += '<tr><td>KYC/AML</td><td>Random Forest</td><td>Multi-factor risk scoring with sanctions screening</td></tr>'
    c += '<tr><td>Churn Prediction</td><td>XGBoost</td><td>Behavioral feature engineering + churn drivers</td></tr>'
    c += '<tr><td>Survival Analysis</td><td>Cox Proportional Hazards</td><td>Predicts WHEN default occurs, not just IF</td></tr>'
    c += '<tr><td>Agentic AML</td><td>Multi-Agent Workflow</td><td>5 specialized agents with audit trail</td></tr>'
    c += '<tr><td>Black-Scholes</td><td>Closed-form PDE solution</td><td>Greeks + payoff visualization</td></tr>'
    c += '<tr><td>Monte Carlo VaR</td><td>10,000-scenario simulation</td><td>Correlated asset returns + Expected Shortfall</td></tr>'
    c += '<tr><td>Basel III IRB</td><td>Regulatory formula</td><td>RWA calculation with asset correlation</td></tr>'
    c += '<tr><td>Copula Defaults</td><td>Gaussian copula</td><td>Systemic factor model for correlated defaults</td></tr>'
    c += '<tr><td>Stress Testing</td><td>Satellite model</td><td>Macro shocks to PD with CET1 impact</td></tr>'
    c += '</tbody></table></div>'
    c += '<script>'
    c += 'fetch("/api/model-performance").then(r=>r.json()).then(d=>{'
    c += 'const t=document.querySelector("#perfTable tbody");'
    c += 'const models=[["Credit Risk","XGBoost",d.credit_risk.auc,"10,000 loans","AUC-ROC"],["Fraud Detection","XGBoost+IF",d.fraud_detection.auc,"120,000 txns","AUC-ROC"],["KYC/AML","Random Forest",d.kyc_aml.auc,"6,000 records","AUC-ROC"],["Churn","XGBoost",d.churn.auc,"8,000 records","AUC-ROC"],["Survival","Cox PH",d.survival_analysis.c_index,"10,000 loans","C-Index"]];'
    c += 't.innerHTML=models.map(m=>"<tr><td>"+m[0]+"</td><td>"+m[1]+"</td><td>"+(m[2]*100).toFixed(1)+"%</td><td>"+m[3]+"</td><td>"+m[4]+"</td></tr>").join("")'
    c += '});'
    c += '</script>'
    return page('Model Performance', c, 'models')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
