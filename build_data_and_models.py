"""
Banking AI Platform - Data Generation & Model Training Pipeline
Generates realistic synthetic banking data and trains ML models for:
  1. Credit Risk Scoring (Retail Banking)
  2. Fraud Detection (Retail / Payments)
  3. KYC/AML Risk Rating (Commercial Banking)
  4. Investment Portfolio Advisor (Investment Banking)
  5. Churn Prediction (Retail Banking)
"""
import os, json, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
random.seed(42)

BASE = Path("/workspace/outputs/banking_ai_app/data")
MODEL_DIR = Path("/workspace/outputs/banking_ai_app/models")
BASE.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 8000
N_TXNS = 120000
N_LOANS = 10000
N_PORTFOLIOS = 6000

print("=" * 60)
print("BANKING AI PLATFORM - DATA & MODEL PIPELINE")
print("=" * 60)

# ------------------------------------------------------------------
# 1. CUSTOMER PROFILES
# ------------------------------------------------------------------
print("\n[1/7] Generating customer profiles...")
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
          'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
          'Surat', 'Kochi', 'Indore', 'Bhopal', 'Chandigarh']
segments = ['Retail', 'Mass Affluent', 'HNWI', 'Corporate', 'SME']
emp_types = ['Salaried', 'Self-Employed', 'Business Owner', 'Retired', 'Student']

customers = pd.DataFrame({
    'customer_id': [f'CUST{100000+i}' for i in range(N_CUSTOMERS)],
    'age': RNG.integers(21, 75, N_CUSTOMERS),
    'gender': RNG.choice(['M', 'F'], N_CUSTOMERS),
    'city': RNG.choice(cities, N_CUSTOMERS),
    'segment': RNG.choice(segments, N_CUSTOMERS, p=[0.45, 0.25, 0.10, 0.12, 0.08]),
    'employment_type': RNG.choice(emp_types, N_CUSTOMERS, p=[0.55, 0.20, 0.12, 0.08, 0.05]),
    'annual_income': np.round(RNG.lognormal(mean=12.5, sigma=0.6, size=N_CUSTOMERS), -2).clip(250000, 50000000),
    'months_on_book': RNG.integers(6, 240, N_CUSTOMERS),
    'num_products': RNG.choice([1,2,3,4,5], N_CUSTOMERS, p=[0.30,0.30,0.20,0.12,0.08]),
})
customers['income_log'] = np.log1p(customers['annual_income'])
customers.to_csv(BASE / "customers.csv", index=False)
print(f"   ✓ {len(customers)} customers generated")

# ------------------------------------------------------------------
# 2. TRANSACTION DATA (for fraud detection)
# ------------------------------------------------------------------
print("[2/7] Generating transaction data...")
txn_types = ['POS', 'ATM Withdrawal', 'Online Transfer', 'UPI Payment',
             'NEFT', 'RTGS', 'Cheque', 'International']
channels = ['Mobile App', 'Internet Banking', 'Branch', 'ATM', 'POS Terminal']
merchants = ['Amazon', 'Flipkart', 'Big Bazaar', 'DMart', 'Reliance', 'Myntra',
             'Swiggy', 'Zomato', 'IRCTC', 'Petrol Pump', 'Hospital', 'Pharmacy',
             'Jeweller', 'Electronics Store', 'Unknown Merchant']

# Base transactions
txns = pd.DataFrame({
    'txn_id': [f'TXN{1000000+i}' for i in range(N_TXNS)],
    'customer_id': np.random.choice(customers['customer_id'].values, N_TXNS),
    'amount': np.round(np.abs(RNG.lognormal(6, 1.8, N_TXNS)), 2).clip(10, 2000000),
    'txn_type': RNG.choice(txn_types, N_TXNS),
    'channel': RNG.choice(channels, N_TXNS),
    'merchant': RNG.choice(merchants, N_TXNS),
    'hour_of_day': RNG.integers(0, 24, N_TXNS),
    'day_of_week': RNG.integers(0, 7, N_TXNS),
    'is_international': RNG.choice([0,1], N_TXNS, p=[0.92, 0.08]),
})
# Derived features
txns['amount_log'] = np.log1p(txns['amount'])
txns['is_night_txn'] = ((txns['hour_of_day'] < 6) | (txns['hour_of_day'] > 22)).astype(int)
txns['is_weekend'] = (txns['day_of_week'] >= 5).astype(int)
txns['is_high_amount'] = (txns['amount'] > 50000).astype(int)

# Fraud label (inject realistic fraud patterns ~3.5%)
fraud_mask = (
    ((txns['is_night_txn'] == 1) & (txns['amount'] > 10000) & (RNG.random(N_TXNS) < 0.25)) |
    ((txns['is_international'] == 1) & (txns['amount'] > 5000) & (RNG.random(N_TXNS) < 0.30)) |
    ((txns['merchant'] == 'Unknown Merchant') & (RNG.random(N_TXNS) < 0.35)) |
    ((txns['amount'] > 100000) & (RNG.random(N_TXNS) < 0.12)) |
    ((txns['txn_type'] == 'International') & (RNG.random(N_TXNS) < 0.20))
)
txns['is_fraud'] = fraud_mask.astype(int)
print(f"   ✓ {len(txns)} transactions generated | Fraud rate: {txns['is_fraud'].mean():.2%}")
txns.to_csv(BASE / "transactions.csv", index=False)

# ------------------------------------------------------------------
# 3. LOAN DATA (for credit risk scoring)
# ------------------------------------------------------------------
print("[3/7] Generating loan application data...")
loan_types = ['Home Loan', 'Personal Loan', 'Auto Loan', 'Education Loan', 'Business Loan']
loan_purposes = ['Home Purchase', 'Home Renovation', 'Vehicle Purchase',
                 'Medical Expense', 'Debt Consolidation', 'Business Expansion',
                 'Education', 'Wedding', 'Travel']

loans = pd.DataFrame({
    'loan_id': [f'LOAN{200000+i}' for i in range(N_LOANS)],
    'customer_id': np.random.choice(customers['customer_id'].values, N_LOANS),
    'loan_type': RNG.choice(loan_types, N_LOANS),
    'loan_amount': np.round(np.abs(RNG.lognormal(12, 0.8, N_LOANS)), -3).clip(50000, 50000000),
    'interest_rate': np.round(RNG.uniform(8.5, 18.5, N_LOANS), 2),
    'tenure_months': RNG.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], N_LOANS),
    'loan_purpose': RNG.choice(loan_purposes, N_LOANS),
})
# Merge customer features
loans = loans.merge(customers[['customer_id','age','annual_income','employment_type',
                                 'months_on_book','num_products','segment']],
                    on='customer_id', how='left')

# Engineered features
loans['loan_to_income'] = loans['loan_amount'] / loans['annual_income']
loans['emi_to_income'] = (loans['loan_amount'] / loans['tenure_months']) / (loans['annual_income']/12)
loans['income_log'] = np.log1p(loans['annual_income'])
loans['loan_amount_log'] = np.log1p(loans['loan_amount'])

# Credit score generation (correlated with income & stability, spread 300-900)
# Normalized income factor: higher income -> higher score, but with noise
inc_norm = (loans['income_log'] - loans['income_log'].min()) / (loans['income_log'].max() - loans['income_log'].min())
base_score = 380 + inc_norm * 380 + loans['months_on_book'] * 0.4 \
             - loans['loan_to_income'] * 25 + RNG.normal(0, 60, N_LOANS)
loans['credit_bureau_score'] = np.clip(base_score.round(), 300, 900).astype(int)

# Default label - tuned for ~12-15% default rate
lgi = np.log1p(loans['annual_income'])   # ~12-18
lti = loans['loan_to_income'].clip(0, 10)  # debt-to-income
cbs = loans['credit_bureau_score']        # 300-900
# Intercept calibrated so logit mean ~ -2.5 (=> ~12% default rate)
logit = (1.8
         + 0.30 * lti
         - 0.007 * cbs
         + 0.06 * loans['interest_rate']
         - 0.15 * lgi
         + 0.015 * (loans['tenure_months'] / 12.0)
         + RNG.normal(0, 0.5, N_LOANS))
default_prob = 1 / (1 + np.exp(-logit))
loans['default'] = (RNG.random(N_LOANS) < default_prob).astype(int)
print(f"   ✓ {len(loans)} loan applications | Default rate: {loans['default'].mean():.2%}")
loans.to_csv(BASE / "loans.csv", index=False)

# ------------------------------------------------------------------
# 4. KYC/AML CUSTOMER RISK DATA
# ------------------------------------------------------------------
print("[4/7] Generating KYC/AML risk data...")
n_aml = 6000
countries = ['India', 'USA', 'UK', 'Singapore', 'UAE', 'Switzerland',
             'Hong Kong', 'Germany', 'Cyprus', 'BVI', 'Panama', 'Seychelles',
             'Mauritius', 'Cayman Islands', 'China']
risk_countries = {'Cyprus', 'BVI', 'Panama', 'Seychelles', 'Mauritius', 'Cayman Islands'}
doc_types = ['PAN', 'Passport', 'Aadhaar', 'Driving License', 'Voter ID']

kyc = pd.DataFrame({
    'entity_id': [f'ENT{300000+i}' for i in range(n_aml)],
    'entity_type': RNG.choice(['Individual', 'Corporate', 'Partnership', 'Trust'], n_aml,
                              p=[0.50, 0.35, 0.10, 0.05]),
    'country': RNG.choice(countries, n_aml),
    'doc_type': RNG.choice(doc_types, n_aml),
    'doc_verified': RNG.choice([0,1], n_aml, p=[0.08, 0.92]),
    'num_bank_accounts': RNG.integers(1, 8, n_aml),
    'txn_volume_monthly': np.round(np.abs(RNG.lognormal(11, 1.2, n_aml)), -2).clip(1000, 100000000),
    'has_pep_link': RNG.choice([0,1], n_aml, p=[0.95, 0.05]),
    'has_sanction_link': RNG.choice([0,1], n_aml, p=[0.98, 0.02]),
    'adverse_media_hits': RNG.choice([0,1,2,3], n_aml, p=[0.80, 0.12, 0.05, 0.03]),
    'years_in_business': RNG.integers(1, 30, n_aml),
    'ownership_transparency': RNG.choice(['Low','Medium','High'], n_aml, p=[0.15,0.35,0.50]),
})
kyc['txn_volume_log'] = np.log1p(kyc['txn_volume_monthly'])
kyc['high_risk_country'] = kyc['country'].isin(risk_countries).astype(int)

# AML risk score: high when sanctions/PEP/adverse media/high-risk country/low transparency
risk_score = (
    kyc['has_sanction_link'] * 40 +
    kyc['has_pep_link'] * 15 +
    kyc['adverse_media_hits'] * 8 +
    kyc['high_risk_country'] * 12 +
    (kyc['ownership_transparency'] == 'Low').astype(int) * 6 +
    (kyc['doc_verified'] == 0).astype(int) * 5 +
    np.clip(kyc['txn_volume_log'] - 12, 0, 10) * 0.3
)
kyc['risk_score'] = np.clip(risk_score.round().astype(int), 0, 100)
kyc['risk_category'] = pd.cut(kyc['risk_score'], bins=[-1, 20, 40, 60, 100],
                               labels=['Low', 'Medium', 'High', 'Critical'])
# Alert flag: High/Critical or any sanctions/PEP
kyc['alert_flag'] = ((kyc['risk_score'] > 40) | (kyc['has_sanction_link'] == 1) |
                     (kyc['has_pep_link'] == 1)).astype(int)
print(f"   ✓ {len(kyc)} KYC entities | Alert rate: {kyc['alert_flag'].mean():.2%}")
kyc.to_csv(BASE / "kyc_aml.csv", index=False)

# ------------------------------------------------------------------
# 5. PORTFOLIO DATA (for investment advisory)
# ------------------------------------------------------------------
print("[5/7] Generating investment portfolio data...")
asset_classes = ['Equity_LargeCap', 'Equity_MidCap', 'Equity_SmallCap',
                 'Debt_Govt', 'Debt_Corporate', 'Gold', 'International',
                 'REIT', 'Commodities', 'Cash']
# Expected returns & volatilities (annualized, realistic for Indian markets)
asset_stats = {
    'Equity_LargeCap':  (0.14, 0.18),
    'Equity_MidCap':     (0.16, 0.24),
    'Equity_SmallCap':   (0.18, 0.30),
    'Debt_Govt':         (0.065, 0.05),
    'Debt_Corporate':    (0.08, 0.07),
    'Gold':              (0.09, 0.14),
    'International':     (0.10, 0.16),
    'REIT':              (0.085, 0.12),
    'Commodities':       (0.07, 0.20),
    'Cash':              (0.035, 0.01),
}

risk_profiles = ['Conservative', 'Moderate', 'Aggressive', 'Very Aggressive']
n_port = N_PORTFOLIOS

# Random allocation weights (summing to 1 across asset classes)
alloc = RNG.dirichlet(np.ones(10) * 0.5, size=n_port)
portfolios = pd.DataFrame(alloc, columns=asset_classes)
portfolios['portfolio_id'] = [f'PORT{400000+i}' for i in range(n_port)]
portfolios['customer_id'] = np.random.choice(customers['customer_id'].values, n_port)
portfolios['risk_profile'] = RNG.choice(risk_profiles, n_port, p=[0.25, 0.35, 0.25, 0.15])
portfolios['investment_horizon_years'] = RNG.choice([1,2,3,5,7,10,15,20], n_port)

# Calculate returns & volatility
returns = np.zeros(n_port)
volatility = np.zeros(n_port)
for i in range(n_port):
    w = alloc[i]
    r = np.array([asset_stats[a][0] for a in asset_classes])
    v = np.array([asset_stats[a][1] for a in asset_classes])
    # Simple correlation assumption
    corr = np.eye(10) * 0.5 + 0.5  # moderate correlation
    cov = np.outer(v, v) * corr
    returns[i] = np.dot(w, r)
    volatility[i] = np.sqrt(w @ cov @ w)

portfolios['expected_return'] = np.round(returns * 100, 2)
portfolios['expected_volatility'] = np.round(volatility * 100, 2)
portfolios['sharpe_ratio'] = np.round(
    (portfolios['expected_return'] - 3.5) / portfolios['expected_volatility'], 3
)
print(f"   ✓ {len(portfolios)} portfolios generated")
portfolios.to_csv(BASE / "portfolios.csv", index=False)

# Save asset stats for the advisor
with open(BASE / "asset_stats.json", "w") as f:
    json.dump(asset_stats, f, indent=2)

# ------------------------------------------------------------------
# 6. CHURN DATA
# ------------------------------------------------------------------
print("[6/7] Generating customer churn data...")
churn_data = customers.copy()
churn_data['avg_monthly_balance'] = np.round(
    RNG.lognormal(9.5, 0.8, len(churn_data)), -2).clip(500, 5000000)
churn_data['num_active_products'] = churn_data['num_products']
churn_data['digital_engagement_score'] = RNG.integers(0, 100, len(churn_data))
churn_data['complaints_last_year'] = RNG.choice([0,0,0,1,1,2,3], len(churn_data))
churn_data['num_branch_visits'] = RNG.integers(0, 20, len(churn_data))
churn_data['credit_card_usage'] = RNG.integers(0, 100, len(churn_data))

# Churn probability
churn_prob = 1 / (1 + np.exp(
    -(-1.5 - 0.03 * churn_data['digital_engagement_score']
      + 0.6 * churn_data['complaints_last_year']
      - 0.15 * churn_data['num_active_products']
      + 0.06 * churn_data['num_branch_visits']
      - 0.2 * np.log1p(churn_data['avg_monthly_balance'])
      + RNG.normal(0, 0.5, len(churn_data)))
))
churn_data['churn'] = (RNG.random(len(churn_data)) < churn_prob).astype(int)
print(f"   ✓ {len(churn_data)} churn records | Churn rate: {churn_data['churn'].mean():.2%}")
churn_data.to_csv(BASE / "churn.csv", index=False)

# ------------------------------------------------------------------
# 7. SUMMARIZE
# ------------------------------------------------------------------
print("[7/7] Data generation complete!")
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"  customers.csv:     {len(customers):,} rows")
print(f"  transactions.csv:  {len(txns):,} rows")
print(f"  loans.csv:         {len(loans):,} rows")
print(f"  kyc_aml.csv:       {len(kyc):,} rows")
print(f"  portfolios.csv:    {len(portfolios):,} rows")
print(f"  churn.csv:         {len(churn_data):,} rows")
print(f"\nAll data saved to: {BASE}")
print("=" * 60)
