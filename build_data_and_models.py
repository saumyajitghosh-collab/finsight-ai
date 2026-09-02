"""
FinSight AI v2.0 - Data Generation & Model Training
Generates synthetic banking data and trains all models:
- Credit Risk (XGBoost)
- Fraud Detection (XGBoost + Autoencoder)
- KYC/AML (Random Forest)
- Churn (XGBoost)
- Survival Analysis (Cox Proportional Hazards)
- Agentic AML (simulated multi-agent workflow data)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

# Lifelines for survival analysis
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except:
    LIFELINES_AVAILABLE = False

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ============ DATA GENERATION ============

def generate_customers(n=8000):
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata',
              'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Kochi']
    segments = ['Retail', 'Premium', 'HNI', 'Mass']
    occupations = ['Salaried', 'Self-Employed', 'Business', 'Professional', 'Retired']
    
    df = pd.DataFrame({
        'customer_id': [f'C{100000+i}' for i in range(n)],
        'age': np.random.normal(38, 12, n).clip(18, 75).astype(int),
        'annual_income': np.random.lognormal(12.8, 0.6, n).clip(150000, 50000000).astype(int),
        'credit_score': np.random.normal(680, 80, n).clip(300, 850).astype(int),
        'city': np.random.choice(cities, n),
        'segment': np.random.choice(segments, n, p=[0.5, 0.25, 0.1, 0.15]),
        'occupation': np.random.choice(occupations, n),
        'months_with_bank': np.random.randint(1, 180, n),
        'num_products': np.random.poisson(2, n).clip(1, 8),
        'avg_monthly_balance': np.random.lognormal(10.5, 0.8, n).clip(500, 5000000).astype(int),
    })
    df['default_flag'] = ((df['credit_score'] < 600) & (df['annual_income'] < 800000) & 
                          (np.random.random(n) < 0.35)).astype(int)
    df['default_flag'] |= ((df['credit_score'] < 550) & (np.random.random(n) < 0.5)).astype(int)
    df['default_flag'] |= ((df['age'] < 25) & (df['annual_income'] < 500000) & 
                           (np.random.random(n) < 0.2)).astype(int)
    high_credit_mask = df['credit_score'] > 750
    df.loc[high_credit_mask, 'default_flag'] = df.loc[high_credit_mask, 'default_flag'] * (np.random.random(high_credit_mask.sum()) < 0.02)
    return df

def generate_transactions(customers, n=120000):
    txns = []
    for _ in range(n):
        cust = customers.sample(1).iloc[0]
        txn_type = np.random.choice(['NEFT', 'IMPS', 'RTGS', 'UPI', 'Card', 'Cheque'], p=[0.2, 0.25, 0.05, 0.35, 0.12, 0.03])
        hour_probs = [0.02,0.01,0.005,0.005,0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,
                      0.06,0.05,0.04,0.04,0.05,0.05,0.06,0.07,0.08,0.06,0.04,0.02]
        hour_probs = np.array(hour_probs) / sum(hour_probs)
        hour = np.random.choice(range(24), p=hour_probs)
        
        base_amt = {'NEFT': 15000, 'IMPS': 5000, 'RTGS': 200000, 'UPI': 2000, 'Card': 8000, 'Cheque': 30000}
        amount = np.random.lognormal(np.log(base_amt[txn_type]), 0.7)
        
        fraud = 0
        if np.random.random() < 0.04:
            fraud = 1
            amount *= np.random.uniform(3, 15)
            hour = np.random.choice([0,1,2,3,22,23])
        
        txns.append({
            'txn_id': f'T{1000000+len(txns)}',
            'customer_id': cust['customer_id'],
            'txn_type': txn_type,
            'amount': round(amount, 2),
            'hour': hour,
            'is_fraud': fraud,
            'channel': np.random.choice(['Mobile', 'NetBanking', 'ATM', 'Branch', 'POS']),
            'merchant_category': np.random.choice(['Retail', 'Food', 'Travel', 'Fuel', 'Entertainment', 'Bills', 'Transfer', 'Other']),
            'location': cust['city'],
            'txn_speed': np.random.exponential(2),
            'device_change': int(np.random.random() < (0.15 if fraud else 0.03)),
            'freq_last_24h': np.random.poisson(3 if not fraud else 8),
        })
    return pd.DataFrame(txns)

def generate_loans(customers, n=10000):
    loans = []
    for _ in range(n):
        cust = customers.sample(1).iloc[0]
        loan_type = np.random.choice(['Home', 'Auto', 'Personal', 'Education', 'Business'])
        base_amt = {'Home': 3500000, 'Auto': 800000, 'Personal': 300000, 'Education': 800000, 'Business': 2500000}
        amount = np.random.lognormal(np.log(base_amt[loan_type]), 0.3)
        rate = 0.08 + np.random.uniform(0, 0.07)
        term = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360])
        credit_score = cust['credit_score']
        
        pd_val = 0.01 + max(0, (650 - credit_score) / 350) * 0.15
        if credit_score < 550:
            pd_val += 0.10
        if cust['annual_income'] < 500000:
            pd_val += 0.05
        
        defaulted = np.random.random() < pd_val
        lgd = np.random.uniform(0.2, 0.9) if defaulted else 0
        ead = amount * np.random.uniform(0.3, 1.0)
        
        # Survival data: months to default or censoring
        if defaulted:
            # Default happens somewhere in the loan term
            months_to_default = int(np.random.exponential(24)) + 3
            months_to_default = min(months_to_default, term)
            event = 1
            observed_time = months_to_default
        else:
            # Censored: loan either paid off or still active
            observed_time = min(np.random.randint(6, term + 12), 360)
            event = 0
        
        loans.append({
            'loan_id': f'L{200000+len(loans)}',
            'customer_id': cust['customer_id'],
            'loan_type': loan_type,
            'loan_amount': round(amount, 2),
            'interest_rate': round(rate, 4),
            'term_months': term,
            'credit_score': credit_score,
            'annual_income': cust['annual_income'],
            'age': cust['age'],
            'dti_ratio': round(np.random.uniform(0.1, 0.6), 3),
            'employment_years': np.random.randint(0, 30),
            'num_dependents': np.random.randint(0, 5),
            'num_prior_loans': np.random.poisson(1.5),
            'has_mortgage': int(np.random.random() < 0.3),
            'months_to_event': observed_time,
            'event': event,  # 1=default, 0=censored
            'defaulted': int(defaulted),
            'lgd': round(lgd, 3),
            'ead': round(ead, 2),
        })
    return pd.DataFrame(loans)

def generate_kyc(customers, n=6000):
    alerts = []
    risk_cats = ['Low', 'Medium', 'High', 'Very High']
    countries = ['India', 'USA', 'UK', 'Singapore', 'UAE', 'Switzerland', 'Cyprus', 'Cayman Is.', 'British Virgin Is.', 'Panama']
    high_risk_countries = ['Cayman Is.', 'British Virgin Is.', 'Panama', 'Cyprus']
    
    for _ in range(n):
        cust = customers.sample(1).iloc[0]
        country = np.random.choice(countries, p=[0.7, 0.08, 0.05, 0.04, 0.04, 0.02, 0.02, 0.02, 0.02, 0.01])
        
        txn_volume = np.random.lognormal(11, 1.5)
        if country in high_risk_countries:
            txn_volume *= 3
        
        num_large_txns = np.random.poisson(3 if country not in high_risk_countries else 8)
        structuring_flag = int(np.random.random() < 0.08)
        pep_flag = int(np.random.random() < 0.02)
        sanctions_hit = int(np.random.random() < 0.01)
        
        risk_score = 0
        if country in high_risk_countries:
            risk_score += 30
        if txn_volume > 5000000:
            risk_score += 25
        if structuring_flag:
            risk_score += 20
        if pep_flag:
            risk_score += 15
        if sanctions_hit:
            risk_score += 50
        if num_large_txns > 5:
            risk_score += 10
        risk_score += np.random.randint(-5, 6)
        risk_score = max(0, min(100, risk_score))
        
        is_suspicious = 1 if risk_score > 50 else 0
        
        alerts.append({
            'kyc_id': f'K{300000+len(alerts)}',
            'customer_id': cust['customer_id'],
            'customer_name': f"Cust_{cust['customer_id']}",
            'country': country,
            'txn_volume_30d': round(txn_volume, 2),
            'num_large_txns': num_large_txns,
            'structuring_detected': structuring_flag,
            'pep_flag': pep_flag,
            'sanctions_hit': sanctions_hit,
            'risk_score': risk_score,
            'risk_category': 'High' if risk_score > 60 else ('Medium' if risk_score > 30 else 'Low'),
            'is_suspicious': is_suspicious,
            'account_age_months': cust['months_with_bank'],
        })
    return pd.DataFrame(alerts)

def generate_portfolios(customers, n=6000):
    ports = []
    risk_profiles = ['Conservative', 'Moderate', 'Aggressive', 'Very Aggressive']
    
    for _ in range(n):
        cust = customers.sample(1).iloc[0]
        profile = np.random.choice(risk_profiles, p=[0.3, 0.4, 0.2, 0.1])
        portfolio_value = np.random.lognormal(12, 0.8)
        
        if profile == 'Conservative':
            equity_pct, debt_pct, gold_pct, cash_pct = 0.2, 0.5, 0.15, 0.15
        elif profile == 'Moderate':
            equity_pct, debt_pct, gold_pct, cash_pct = 0.45, 0.35, 0.1, 0.1
        elif profile == 'Aggressive':
            equity_pct, debt_pct, gold_pct, cash_pct = 0.65, 0.2, 0.1, 0.05
        else:
            equity_pct, debt_pct, gold_pct, cash_pct = 0.8, 0.1, 0.05, 0.05
        
        # Simulate annual return based on profile
        base_return = {'Conservative': 0.07, 'Moderate': 0.10, 'Aggressive': 0.13, 'Very Aggressive': 0.16}
        actual_return = np.random.normal(base_return[profile], 0.08)
        
        benchmark_return = 0.09
        alpha = actual_return - benchmark_return * equity_pct - 0.06 * debt_pct - 0.08 * gold_pct - 0.04 * cash_pct
        
        ports.append({
            'portfolio_id': f'P{400000+len(ports)}',
            'customer_id': cust['customer_id'],
            'risk_profile': profile,
            'portfolio_value': round(portfolio_value, 2),
            'equity_pct': equity_pct,
            'debt_pct': debt_pct,
            'gold_pct': gold_pct,
            'cash_pct': cash_pct,
            'annual_return': round(actual_return * 100, 2),
            'benchmark_return': 9.0,
            'alpha': round(alpha * 100, 2),
            'sharpe_ratio': round(actual_return / 0.08, 3),
            'volatility': round(np.random.uniform(8, 25), 2),
            'max_drawdown': round(np.random.uniform(-30, -5), 2),
            'age': cust['age'],
            'income': cust['annual_income'],
        })
    return pd.DataFrame(ports)

def generate_churn(customers, n=8000):
    df = customers.copy()
    df['satisfaction_score'] = np.random.uniform(2, 5, n).round(2)
    df['complaints_last_6m'] = np.random.poisson(1.5, n)
    df['digital_engagement'] = np.random.uniform(0, 1, n).round(3)
    df['branch_visits_3m'] = np.random.poisson(3, n)
    df['product_utilization'] = np.random.uniform(0.1, 1.0, n).round(3)
    
    churn_prob = 0.05
    churn_prob += (df['satisfaction_score'] < 3) * 0.20
    churn_prob += (df['complaints_last_6m'] > 3) * 0.15
    churn_prob += (df['digital_engagement'] < 0.2) * 0.10
    churn_prob += (df['months_with_bank'] < 12) * 0.08
    churn_prob = churn_prob.clip(0, 0.6)
    
    df['churned'] = (np.random.random(n) < churn_prob).astype(int)
    return df

# ============ MODEL TRAINING ============

def train_credit_risk_model(loans):
    le_loan = LabelEncoder()
    loans['loan_type_enc'] = le_loan.fit_transform(loans['loan_type'])
    
    features = ['loan_amount', 'interest_rate', 'term_months', 'credit_score', 
                'annual_income', 'age', 'dti_ratio', 'employment_years', 
                'num_dependents', 'num_prior_loans', 'has_mortgage', 'loan_type_enc']
    
    X = loans[features]
    y = loans['defaulted']
    
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                               use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"Credit Risk AUC: {auc:.4f}")
    
    with open(MODELS_DIR / 'credit_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': features, 'encoder': le_loan, 'auc': auc}, f)
    return model, features, auc

def train_fraud_model(txns):
    le_type = LabelEncoder()
    le_channel = LabelEncoder()
    le_merchant = LabelEncoder()
    le_location = LabelEncoder()
    
    txns['txn_type_enc'] = le_type.fit_transform(txns['txn_type'])
    txns['channel_enc'] = le_channel.fit_transform(txns['channel'])
    txns['merchant_enc'] = le_merchant.fit_transform(txns['merchant_category'])
    txns['location_enc'] = le_location.fit_transform(txns['location'])
    
    features = ['amount', 'hour', 'txn_type_enc', 'channel_enc', 'merchant_enc', 
                'location_enc', 'txn_speed', 'device_change', 'freq_last_24h']
    
    X = txns[features]
    y = txns['is_fraud']
    
    model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"Fraud Detection AUC: {auc:.4f}")
    
    # Also train Autoencoder for anomaly detection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Isolation Forest as autoencoder substitute (works without tensorflow)
    iso_forest = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    iso_forest.fit(X_scaled)
    
    with open(MODELS_DIR / 'fraud_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 
            'features': features, 
            'encoders': {'txn_type': le_type, 'channel': le_channel, 
                        'merchant': le_merchant, 'location': le_location},
            'scaler': scaler,
            'iso_forest': iso_forest,
            'auc': auc
        }, f)
    return model, features, auc

def train_kyc_model(kyc):
    le_country = LabelEncoder()
    le_risk_cat = LabelEncoder()
    kyc['country_enc'] = le_country.fit_transform(kyc['country'])
    kyc['risk_cat_enc'] = le_risk_cat.fit_transform(kyc['risk_category'])
    
    features = ['country_enc', 'txn_volume_30d', 'num_large_txns', 'structuring_detected',
                'pep_flag', 'sanctions_hit', 'risk_score', 'account_age_months']
    
    X = kyc[features]
    y = kyc['is_suspicious']
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"KYC/AML AUC: {auc:.4f}")
    
    with open(MODELS_DIR / 'kyc_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': features, 'encoder': le_country, 'auc': auc}, f)
    return model, features, auc

def train_churn_model(churn_df):
    le_segment = LabelEncoder()
    le_occupation = LabelEncoder()
    le_city = LabelEncoder()
    churn_df['segment_enc'] = le_segment.fit_transform(churn_df['segment'])
    churn_df['occupation_enc'] = le_occupation.fit_transform(churn_df['occupation'])
    churn_df['city_enc'] = le_city.fit_transform(churn_df['city'])
    
    features = ['age', 'annual_income', 'credit_score', 'months_with_bank', 'num_products',
                'avg_monthly_balance', 'satisfaction_score', 'complaints_last_6m',
                'digital_engagement', 'branch_visits_3m', 'product_utilization',
                'segment_enc', 'occupation_enc', 'city_enc']
    
    X = churn_df[features]
    y = churn_df['churned']
    
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"Churn AUC: {auc:.4f}")
    
    with open(MODELS_DIR / 'churn_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 'features': features, 
            'encoders': {'segment': le_segment, 'occupation': le_occupation, 'city': le_city},
            'auc': auc
        }, f)
    return model, features, auc

def train_survival_model(loans):
    """Train Cox Proportional Hazards model for loan default timing."""
    if not LIFELINES_AVAILABLE:
        print("Lifelines not available, skipping survival model")
        return None, None, 0
    
    features = ['loan_amount', 'interest_rate', 'term_months', 'credit_score',
                'annual_income', 'age', 'dti_ratio', 'employment_years',
                'num_dependents', 'num_prior_loans', 'has_mortgage']
    
    # Add loan_type as categorical
    loan_type_dummies = pd.get_dummies(loans['loan_type'], prefix='loan_type', drop_first=True)
    
    cox_data = pd.concat([loans[features], loan_type_dummies], axis=1)
    cox_data['duration'] = loans['months_to_event']
    cox_data['event'] = loans['event']
    
    # Ensure no inf or nan
    cox_data = cox_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_data, duration_col='duration', event_col='event')
    
    c_index = concordance_index(cox_data['duration'], -cph.predict_partial_hazard(cox_data), cox_data['event'])
    print(f"Survival Analysis C-Index: {c_index:.4f}")
    
    # Extract hazard ratios
    hazard_ratios = np.exp(cph.params_)
    
    with open(MODELS_DIR / 'survival_model.pkl', 'wb') as f:
        pickle.dump({
            'model': cph,
            'features': features,
            'dummy_columns': list(loan_type_dummies.columns),
            'c_index': c_index,
            'hazard_ratios': hazard_ratios.to_dict(),
            'summary': cph.summary
        }, f)
    
    return cph, features, c_index


# ============ MAIN ============

if __name__ == '__main__':
    print("=" * 60)
    print("FinSight AI v2.0 - Building Data & Models")
    print("=" * 60)
    
    print("\n1. Generating Customers...")
    customers = generate_customers(8000)
    customers.to_csv(DATA_DIR / 'customers.csv', index=False)
    print(f"   {len(customers)} customers generated")
    
    print("\n2. Generating Transactions...")
    transactions = generate_transactions(customers, 120000)
    transactions.to_csv(DATA_DIR / 'transactions.csv', index=False)
    print(f"   {len(transactions)} transactions generated ({transactions['is_fraud'].sum()} fraud)")
    
    print("\n3. Generating Loans (with survival data)...")
    loans = generate_loans(customers, 10000)
    loans.to_csv(DATA_DIR / 'loans.csv', index=False)
    print(f"   {len(loans)} loans generated ({loans['defaulted'].sum()} defaults, {loans['event'].sum()} events)")
    
    print("\n4. Generating KYC/AML data...")
    kyc = generate_kyc(customers, 6000)
    kyc.to_csv(DATA_DIR / 'kyc.csv', index=False)
    print(f"   {len(kyc)} KYC records generated ({kyc['is_suspicious'].sum()} suspicious)")
    
    print("\n5. Generating Portfolios...")
    portfolios = generate_portfolios(customers, 6000)
    portfolios.to_csv(DATA_DIR / 'portfolios.csv', index=False)
    print(f"   {len(portfolios)} portfolios generated")
    
    print("\n6. Generating Churn data...")
    churn = generate_churn(customers, 8000)
    churn.to_csv(DATA_DIR / 'churn.csv', index=False)
    print(f"   {len(churn)} churn records generated ({churn['churned'].sum()} churned)")
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    print("\n7. Training Credit Risk Model (XGBoost)...")
    train_credit_risk_model(loans)
    
    print("\n8. Training Fraud Detection Model (XGBoost + IsolationForest)...")
    train_fraud_model(transactions)
    
    print("\n9. Training KYC/AML Model (Random Forest)...")
    train_kyc_model(kyc)
    
    print("\n10. Training Churn Model (XGBoost)...")
    train_churn_model(churn)
    
    print("\n11. Training Survival Analysis Model (Cox PH)...")
    train_survival_model(loans)
    
    # Save summary stats for dashboard
    summary = {
        'total_customers': len(customers),
        'total_transactions': len(transactions),
        'total_loans': len(loans),
        'total_fraud_alerts': int(transactions['is_fraud'].sum()),
        'total_defaults': int(loans['defaulted'].sum()),
        'total_kyc_alerts': int(kyc['is_suspicious'].sum()),
        'total_churn': int(churn['churned'].sum()),
        'total_portfolios': len(portfolios),
        'survival_events': int(loans['event'].sum()),
    }
    with open(MODELS_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DATA & MODELS BUILD COMPLETE!")
    print("=" * 60)
    print(f"Data saved to: {DATA_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
