"""
FinSight AI — Banking Intelligence Platform (Single File, No Templates Folder)
All HTML is embedded as Python strings. Only need this one file + data/ + models/
"""
import os, json, gc
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

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
churn_df = None


def _load_models():
    global _credit_model, _fraud_model, _kyc_model, _churn_model
    global _encoders, _credit_features, _fraud_features, _kyc_features, _churn_features
    global _model_metrics, _asset_stats
    if _credit_model is not None:
        return
    print("Loading models...", flush=True)
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
    print("Models loaded.", flush=True)


def _load_data():
    global customers_df, txns_df, loans_df, kyc_df, churn_df
    if customers_df is not None:
        return
    print("Loading data...", flush=True)
    customers_df = pd.read_csv(DATA_DIR / "customers.csv")
    txns_df = pd.read_csv(DATA_DIR / "transactions.csv")
    loans_df = pd.read_csv(DATA_DIR / "loans.csv")
    kyc_df = pd.read_csv(DATA_DIR / "kyc_aml.csv")
    churn_df = pd.read_csv(DATA_DIR / "churn.csv")
    gc.collect()
    print("Data loaded.", flush=True)


def encode_input(df, feature_list, encoder_prefix, encoders):
    df = df.copy()
    for col in feature_list:
        if col in df.columns:
            le_key = encoder_prefix + "_" + col
            if le_key in encoders:
                le = encoders[le_key]
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    return df


# ─── Embedded HTML ──────────────────────────────────────────────
BASE_CSS = """
:root{--primary:#1a237e;--primary-light:#3949ab;--accent:#00bfa5;--danger:#e53935;--warning:#fb8c00;--success:#43a047;--bg:#f5f6fa;--card-bg:#fff;--text:#1a1a2e;--text-muted:#6c757d;--border:#e0e0e0;--radius:12px;--shadow:0 2px 8px rgba(0,0,0,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text)}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:240px;background:linear-gradient(180deg,var(--primary),var(--primary-light));color:#fff;padding:24px 0;overflow-y:auto;z-index:100}
.sidebar-header{text-align:center;padding:0 20px 28px;border-bottom:1px solid rgba(255,255,255,0.1)}
.sidebar-header h1{font-size:1.5rem;font-weight:700}
.sidebar-header p{font-size:0.72rem;opacity:0.7;margin-top:4px}
.sidebar nav{padding:20px 0}
.sidebar nav a{display:flex;align-items:center;gap:12px;padding:12px 24px;color:rgba(255,255,255,0.75);text-decoration:none;font-size:0.9rem;font-weight:500;transition:all 0.2s;border-left:3px solid transparent}
.sidebar nav a:hover{background:rgba(255,255,255,0.1);color:#fff}
.sidebar nav a.active{background:rgba(255,255,255,0.15);color:#fff;border-left-color:var(--accent)}
.sidebar-footer{position:absolute;bottom:16px;left:0;right:0;text-align:center;font-size:0.7rem;opacity:0.5}
.main-content{margin-left:240px;padding:32px;min-height:100vh}
.page-header{margin-bottom:28px}
.page-header h2{font-size:1.7rem;font-weight:700;color:var(--primary)}
.page-header p{color:var(--text-muted);font-size:0.9rem;margin-top:4px}
.card{background:var(--card-bg);border-radius:var(--radius);padding:24px;box-shadow:var(--shadow);margin-bottom:20px}
.card-title{font-size:1.05rem;font-weight:600;margin-bottom:16px;color:var(--primary)}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px}
.kpi-card{background:var(--card-bg);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow);border-left:4px solid var(--accent);transition:transform 0.2s}
.kpi-card:hover{transform:translateY(-2px)}
.kpi-card .label{font-size:0.78rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;font-weight:600}
.kpi-card .value{font-size:1.8rem;font-weight:700;margin:8px 0 4px}
.kpi-card .sub{font-size:0.78rem;color:var(--text-muted)}
.kpi-card.danger{border-left-color:var(--danger)}
.kpi-card.warning{border-left-color:var(--warning)}
.kpi-card.success{border-left-color:var(--success)}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
.chart-container{position:relative;height:280px}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.form-group{margin-bottom:14px}
.form-group label{display:block;font-size:0.82rem;font-weight:600;color:var(--text-muted);margin-bottom:5px}
.form-group input,.form-group select{width:100%;padding:10px 12px;border:1px solid var(--border);border-radius:8px;font-size:0.9rem}
.form-group input:focus,.form-group select:focus{outline:none;border-color:var(--primary-light);box-shadow:0 0 0 3px rgba(57,73,171,0.1)}
.btn{padding:11px 24px;border:none;border-radius:8px;font-size:0.9rem;font-weight:600;cursor:pointer;transition:all 0.2s;display:inline-flex;align-items:center;gap:8px}
.btn-primary{background:var(--primary);color:#fff}
.btn-primary:hover{background:var(--primary-light)}
.btn-secondary{background:#e8eaf6;color:var(--primary)}
.btn-secondary:hover{background:#c5cae9}
.result-box{margin-top:20px;padding:20px;border-radius:var(--radius);display:none}
.result-box.show{display:block;animation:fadeIn 0.3s}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.result-box.approve{background:#e8f5e9;border:1px solid var(--success)}
.result-box.warn{background:#fff3e0;border:1px solid var(--warning)}
.result-box.danger{background:#ffebee;border:1px solid var(--danger)}
.result-box h3{font-size:1.2rem;margin-bottom:10px}
.result-box .prob-bar{height:24px;border-radius:12px;background:#e0e0e0;overflow:hidden;margin:12px 0}
.result-box .prob-fill{height:100%;border-radius:12px;transition:width 0.5s}
table{width:100%;border-collapse:collapse}
table th{text-align:left;font-size:0.78rem;text-transform:uppercase;color:var(--text-muted);padding:10px 12px;border-bottom:2px solid var(--border)}
table td{padding:10px 12px;border-bottom:1px solid var(--border);font-size:0.88rem}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:600}
.badge-success{background:#e8f5e9;color:var(--success)}
.badge-warning{background:#fff3e0;color:var(--warning)}
.badge-danger{background:#ffebee;color:var(--danger)}
.divider{height:1px;background:var(--border);margin:16px 0}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.info-item{display:flex;justify-content:space-between;padding:6px 0;font-size:0.85rem}
.info-item .key{color:var(--text-muted)}
.info-item .val{font-weight:600}
.allocation-bar{display:flex;height:32px;border-radius:8px;overflow:hidden;margin:12px 0}
.alloc-segment{display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:#fff;font-weight:600;overflow:hidden}
.footer-note{text-align:center;padding:20px;font-size:0.8rem;color:var(--text-muted)}
@media(max-width:900px){.chart-grid{grid-template-columns:1fr}.sidebar{width:60px}.sidebar-header h1,.sidebar-header p,.sidebar nav a span:not(.icon),.sidebar-footer{display:none}.main-content{margin-left:60px}}
@media(max-width:700px){.form-grid{grid-template-columns:1fr}}
"""

SIDEBAR = """
<div class="sidebar">
<div class="sidebar-header"><h1>FinSight AI</h1><p>Banking Intelligence Platform</p></div>
<nav>
<a href="/"><span class="icon">📊</span> <span>Dashboard</span></a>
<a href="/credit-risk"><span class="icon">🏦</span> <span>Credit Risk</span></a>
<a href="/fraud-detection"><span class="icon">🛡️</span> <span>Fraud Detection</span></a>
<a href="/kyc-aml"><span class="icon">📋</span> <span>KYC / AML</span></a>
<a href="/portfolio-advisor"><span class="icon">📈</span> <span>Portfolio Advisor</span></a>
<a href="/churn"><span class="icon">👥</span> <span>Churn Prediction</span></a>
<a href="/models"><span class="icon">⚙️</span> <span>Model Performance</span></a>
</nav>
<div class="sidebar-footer">v1.0 · Synthetic Data</div>
</div>
"""


def page(title, subtitle, content_html, script_js=""):
    return '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>FinSight AI</title>\n<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>\n<style>' + BASE_CSS + '</style>\n</head>\n<body>\n' + SIDEBAR + '\n<div class="main-content">\n<div class="page-header"><h2>' + title + '</h2><p>' + subtitle + '</p></div>\n' + content_html + '\n</div>\n<script>' + script_js + '</script>\n</body>\n</html>'


# ─── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    content = """<div class="kpi-grid">
<div class="kpi-card"><div class="label">Total Customers</div><div class="value" id="kpi-customers">—</div><div class="sub" id="kpi-segments">—</div></div>
<div class="kpi-card danger"><div class="label">Fraud Alerts</div><div class="value" id="kpi-fraud">—</div><div class="sub" id="kpi-fraud-rate">—</div></div>
<div class="kpi-card warning"><div class="label">Loan Defaults</div><div class="value" id="kpi-defaults">—</div><div class="sub" id="kpi-default-rate">—</div></div>
<div class="kpi-card"><div class="label">KYC Alerts</div><div class="value" id="kpi-kyc">—</div><div class="sub">AML risk flags</div></div>
<div class="kpi-card warning"><div class="label">Churn Risk</div><div class="value" id="kpi-churn">—</div><div class="sub" id="kpi-churn-rate">—</div></div>
<div class="kpi-card success"><div class="label">Transactions</div><div class="value" id="kpi-txns">—</div><div class="sub">Total processed</div></div>
</div>
<div class="chart-grid">
<div class="card"><div class="card-title">Customer Segment Distribution</div><div class="chart-container"><canvas id="chart-segments"></canvas></div></div>
<div class="card"><div class="card-title">Fraud by Transaction Type</div><div class="chart-container"><canvas id="chart-fraud-type"></canvas></div></div>
</div>
<div class="chart-grid">
<div class="card"><div class="card-title">Loan Type Distribution</div><div class="chart-container"><canvas id="chart-loan-types"></canvas></div></div>
<div class="card"><div class="card-title">KYC Risk Category Distribution</div><div class="chart-container"><canvas id="chart-risk-cat"></canvas></div></div>
</div>
<div class="card"><div class="card-title">Model Performance Summary</div>
<table><thead><tr><th>Model</th><th>Domain</th><th>Algorithm</th><th>AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Train Samples</th></tr></thead>
<tbody id="model-table-body"></tbody></table>
</div>
<div class="footer-note">FinSight AI Platform · Synthetic data · Models: scikit-learn & XGBoost</div>"""

    js = """var COLORS=['#1a237e','#3949ab','#00bfa5','#fb8c00','#e53935','#43a047','#8e24aa','#00897b'];
var algoMap={'credit_risk':'XGBoost','fraud_detection':'XGBoost','kyc_aml':'Random Forest','churn':'XGBoost'};
var domainMap={'credit_risk':'Retail Banking','fraud_detection':'Retail / Payments','kyc_aml':'Commercial Banking','churn':'Retail Banking'};
fetch('/api/dashboard').then(r=>r.json()).then(data=>{
document.getElementById('kpi-customers').textContent=data.total_customers.toLocaleString();
document.getElementById('kpi-segments').textContent=Object.keys(data.segment_distribution).length+' segments';
document.getElementById('kpi-fraud').textContent=data.fraud_count.toLocaleString();
document.getElementById('kpi-fraud-rate').textContent=data.fraud_rate+'% fraud rate';
document.getElementById('kpi-defaults').textContent=data.default_count.toLocaleString();
document.getElementById('kpi-default-rate').textContent=data.default_rate+'% default rate';
document.getElementById('kpi-kyc').textContent=data.kyc_alerts.toLocaleString();
document.getElementById('kpi-txns').textContent=data.total_transactions.toLocaleString();
document.getElementById('kpi-churn').textContent=data.churn_count.toLocaleString();
document.getElementById('kpi-churn-rate').textContent=data.churn_rate+'% churn rate';
new Chart(document.getElementById('chart-segments'),{type:'doughnut',data:{labels:Object.keys(data.segment_distribution),datasets:[{data:Object.values(data.segment_distribution),backgroundColor:COLORS}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right'}}}});
new Chart(document.getElementById('chart-fraud-type'),{type:'bar',data:{labels:Object.keys(data.fraud_by_type),datasets:[{data:Object.values(data.fraud_by_type),backgroundColor:'#e53935'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}}}});
new Chart(document.getElementById('chart-loan-types'),{type:'bar',data:{labels:Object.keys(data.loan_type_distribution),datasets:[{data:Object.values(data.loan_type_distribution),backgroundColor:'#1a237e'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},indexAxis:'y'}});
new Chart(document.getElementById('chart-risk-cat'),{type:'doughnut',data:{labels:Object.keys(data.risk_category_distribution),datasets:[{data:Object.values(data.risk_category_distribution),backgroundColor:['#43a047','#fb8c00','#e53935','#b71c1c']}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right'}}}});
var tbody=document.getElementById('model-table-body');
var keys=Object.keys(data.model_metrics);
for(var i=0;i<keys.length;i++){var key=keys[i];var m=data.model_metrics[key];var name=key.replace(/_/g,' ');
tbody.innerHTML+='<tr><td><strong>'+name+'</strong></td><td>'+domainMap[key]+'</td><td>'+algoMap[key]+'</td><td><span class="badge badge-success">'+m.auc+'</span></td><td>'+m.precision+'</td><td>'+m.recall+'</td><td>'+m.f1+'</td><td>'+m.train_samples.toLocaleString()+'</td></tr>';}
});"""
    return page("Banking Intelligence Dashboard", "Cross-functional AI platform spanning Retail, Commercial, and Investment Banking", content, js)


@app.route('/api/dashboard')
def dashboard():
    _load_models()
    _load_data()
    return jsonify({
        'total_customers': len(customers_df),
        'total_transactions': len(txns_df),
        'fraud_count': int(txns_df['is_fraud'].sum()),
        'fraud_rate': round(int(txns_df['is_fraud'].sum()) / len(txns_df) * 100, 2),
        'total_loans': len(loans_df),
        'default_count': int(loans_df['default'].sum()),
        'default_rate': round(int(loans_df['default'].sum()) / len(loans_df) * 100, 2),
        'kyc_alerts': int(kyc_df['alert_flag'].sum()),
        'churn_count': int(churn_df['churn'].sum()),
        'churn_rate': round(int(churn_df['churn'].sum()) / len(churn_df) * 100, 2),
        'segment_distribution': customers_df['segment'].value_counts().to_dict(),
        'loan_type_distribution': loans_df['loan_type'].value_counts().to_dict(),
        'fraud_by_type': txns_df.groupby('txn_type')['is_fraud'].sum().to_dict(),
        'risk_category_distribution': kyc_df['risk_category'].value_counts().to_dict(),
        'model_metrics': _model_metrics,
    })


@app.route('/credit-risk')
def credit_risk_page():
    content = """<div class="card"><div class="card-title">Loan Application Details</div>
<div class="form-grid">
<div class="form-group"><label>Loan Amount (₹)</label><input type="number" id="loan_amount" value="1500000" step="50000"></div>
<div class="form-group"><label>Interest Rate (%)</label><input type="number" id="interest_rate" value="12.5" step="0.1"></div>
<div class="form-group"><label>Tenure (months)</label><select id="tenure_months"><option value="12">12</option><option value="24">24</option><option value="36">36</option><option value="48">48</option><option value="60">60</option><option value="84">84</option><option value="120">120</option><option value="180" selected>180</option><option value="240">240</option></select></div>
<div class="form-group"><label>Applicant Age</label><input type="number" id="age" value="35"></div>
<div class="form-group"><label>Annual Income (₹)</label><input type="number" id="annual_income" value="1200000" step="50000"></div>
<div class="form-group"><label>Credit Bureau Score (300-900)</label><input type="number" id="credit_bureau_score" value="720" min="300" max="900"></div>
<div class="form-group"><label>Employment Type</label><select id="employment_type"><option value="Salaried">Salaried</option><option value="Self-Employed">Self-Employed</option><option value="Business Owner">Business Owner</option><option value="Retired">Retired</option><option value="Student">Student</option></select></div>
<div class="form-group"><label>Loan Type</label><select id="loan_type"><option value="Home Loan" selected>Home Loan</option><option value="Personal Loan">Personal Loan</option><option value="Auto Loan">Auto Loan</option><option value="Education Loan">Education Loan</option><option value="Business Loan">Business Loan</option></select></div>
<div class="form-group"><label>Customer Segment</label><select id="segment"><option value="Retail" selected>Retail</option><option value="Mass Affluent">Mass Affluent</option><option value="HNWI">HNWI</option><option value="Corporate">Corporate</option><option value="SME">SME</option></select></div>
<div class="form-group"><label>Months with Bank</label><input type="number" id="months_on_book" value="48"></div>
</div>
<div style="display:flex;gap:12px;margin-top:8px;"><button class="btn btn-primary" onclick="predictCredit()">Assess Credit Risk</button><button class="btn btn-secondary" onclick="loadSample()">Load Random Sample</button></div>
</div>
<div class="result-box" id="result"><h3 id="result-title"></h3><div id="result-body"></div></div>
<div class="card" style="margin-top:20px;"><div class="card-title">How It Works</div><p style="font-size:0.88rem;color:var(--text-muted);line-height:1.6;">The Credit Risk model uses XGBoost trained on 10,000 loan applications. The model outputs a default probability mapped to a risk grade (A-E) with an actionable lending recommendation.</p></div>"""

    js = """function val(id){return document.getElementById(id).value;}
function predictCredit(){
var d={loan_amount:val('loan_amount'),interest_rate:val('interest_rate'),tenure_months:val('tenure_months'),age:val('age'),annual_income:val('annual_income'),credit_bureau_score:val('credit_bureau_score'),employment_type:val('employment_type'),loan_type:val('loan_type'),segment:val('segment'),months_on_book:val('months_on_book'),num_products:2};
fetch('/api/credit-risk/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(function(r){return r.json();}).then(function(res){
var box=document.getElementById('result');var cls='approve';
if(res.risk_grade==='D'||res.risk_grade==='E')cls='danger';else if(res.risk_grade==='C')cls='warn';
box.className='result-box show '+cls;
var color=res.risk_grade<='B'?'#43a047':res.risk_grade==='C'?'#fb8c00':'#e53935';
document.getElementById('result-title').innerHTML='Risk Grade: '+res.risk_grade+' — Default Probability: '+res.default_probability+'%';
var fh='';
for(var i=0;i<res.top_risk_factors.length;i++){var f=res.top_risk_factors[i];fh+='<div class="info-item"><span class="key">'+f.feature+'</span><span class="val">'+(f.importance*100).toFixed(1)+'%</span></div>';}
document.getElementById('result-body').innerHTML='<p style="font-size:1rem;font-weight:600;margin-bottom:8px;">'+res.recommendation+'</p><div class="prob-bar"><div class="prob-fill" style="width:'+res.default_probability+'%;background:'+color+';"></div></div><div class="divider"></div><div class="info-grid"><div class="info-item"><span class="key">Loan Amount</span><span class="val">'+res.input_summary.loan_amount+'</span></div><div class="info-item"><span class="key">Loan-to-Income</span><span class="val">'+res.input_summary.loan_to_income+'</span></div><div class="info-item"><span class="key">Credit Score</span><span class="val">'+res.input_summary.credit_score+'</span></div><div class="info-item"><span class="key">Prediction</span><span class="val">'+res.prediction+'</span></div></div><div class="divider"></div><p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:6px;">Top Risk Factors:</p>'+fh;
}).catch(function(err){document.getElementById('result').className='result-box show danger';document.getElementById('result-title').innerHTML='Error';document.getElementById('result-body').innerHTML='<p>'+err.message+'</p>';});
}
function loadSample(){fetch('/api/credit-risk/sample').then(function(r){return r.json();}).then(function(s){document.getElementById('loan_amount').value=s.loan_amount;document.getElementById('interest_rate').value=s.interest_rate;document.getElementById('tenure_months').value=s.tenure_months;document.getElementById('age').value=s.age;document.getElementById('annual_income').value=s.annual_income;document.getElementById('credit_bureau_score').value=s.credit_bureau_score;document.getElementById('employment_type').value=s.employment_type;document.getElementById('loan_type').value=s.loan_type;document.getElementById('segment').value=s.segment;});}"""
    return page("Credit Risk Scoring", "AI-powered loan default prediction using XGBoost", content, js)


@app.route('/api/credit-risk/predict', methods=['POST'])
def credit_risk_predict():
    _load_models()
    data = request.json
    input_data = pd.DataFrame([{
        'loan_amount': float(data['loan_amount']), 'interest_rate': float(data['interest_rate']),
        'tenure_months': int(data['tenure_months']), 'age': int(data['age']),
        'annual_income': float(data['annual_income']), 'months_on_book': int(data.get('months_on_book', 36)),
        'num_products': int(data.get('num_products', 2)),
        'loan_to_income': float(data['loan_amount']) / float(data['annual_income']),
        'emi_to_income': (float(data['loan_amount']) / int(data['tenure_months'])) / (float(data['annual_income']) / 12),
        'income_log': np.log1p(float(data['annual_income'])),
        'loan_amount_log': np.log1p(float(data['loan_amount'])),
        'credit_bureau_score': int(data['credit_bureau_score']),
        'employment_type': data['employment_type'], 'loan_type': data['loan_type'],
        'segment': data.get('segment', 'Retail'),
    }])
    input_encoded = encode_input(input_data, _credit_features, 'credit', _encoders)[_credit_features]
    proba = _credit_model.predict_proba(input_encoded)[0]
    dp = float(proba[1])
    if dp < 0.05: g, rec = 'A', 'APPROVE — Low risk borrower'
    elif dp < 0.15: g, rec = 'B', 'APPROVE — Moderate risk, standard terms'
    elif dp < 0.30: g, rec = 'C', 'CONDITIONAL — Higher interest or collateral required'
    elif dp < 0.50: g, rec = 'D', 'REVIEW — Manual underwriting recommended'
    else: g, rec = 'E', 'DECLINE — High default probability'
    return jsonify({
        'default_probability': round(dp * 100, 2), 'prediction': 'DEFAULT' if dp > 0.5 else 'NO DEFAULT',
        'risk_grade': g, 'recommendation': rec,
        'top_risk_factors': _model_metrics['credit_risk']['feature_importance'][:5],
        'input_summary': {
            'loan_amount': 'Rs ' + format(float(data['loan_amount']), ',.0f'),
            'loan_to_income': format(float(data['loan_amount'])/float(data['annual_income']), '.2f'),
            'credit_score': data['credit_bureau_score'],
        }
    })


@app.route('/api/credit-risk/sample')
def credit_risk_sample():
    _load_data()
    s = loans_df.sample(1).iloc[0]
    return jsonify({'loan_amount': int(s['loan_amount']), 'interest_rate': float(s['interest_rate']),
        'tenure_months': int(s['tenure_months']), 'age': int(s['age']), 'annual_income': int(s['annual_income']),
        'credit_bureau_score': int(s['credit_bureau_score']), 'employment_type': str(s['employment_type']),
        'loan_type': str(s['loan_type']), 'segment': str(s['segment']), 'actual_default': int(s['default'])})


@app.route('/fraud-detection')
def fraud_page():
    content = """<div class="card"><div class="card-title">Transaction Details</div>
<div class="form-grid">
<div class="form-group"><label>Transaction Amount (₹)</label><input type="number" id="amount" value="25000" step="100"></div>
<div class="form-group"><label>Transaction Type</label><select id="txn_type"><option value="POS">POS</option><option value="ATM Withdrawal">ATM Withdrawal</option><option value="Online Transfer">Online Transfer</option><option value="UPI Payment" selected>UPI Payment</option><option value="NEFT">NEFT</option><option value="RTGS">RTGS</option><option value="Cheque">Cheque</option><option value="International">International</option></select></div>
<div class="form-group"><label>Channel</label><select id="channel"><option value="Mobile App" selected>Mobile App</option><option value="Internet Banking">Internet Banking</option><option value="Branch">Branch</option><option value="ATM">ATM</option><option value="POS Terminal">POS Terminal</option></select></div>
<div class="form-group"><label>Merchant</label><select id="merchant"><option value="Amazon">Amazon</option><option value="Flipkart">Flipkart</option><option value="Big Bazaar">Big Bazaar</option><option value="DMart">DMart</option><option value="Reliance">Reliance</option><option value="Swiggy">Swiggy</option><option value="Zomato">Zomato</option><option value="IRCTC">IRCTC</option><option value="Petrol Pump">Petrol Pump</option><option value="Unknown Merchant">Unknown Merchant</option></select></div>
<div class="form-group"><label>Hour of Day (0-23)</label><input type="number" id="hour_of_day" value="14" min="0" max="23"></div>
<div class="form-group"><label>Day of Week (0=Mon, 6=Sun)</label><input type="number" id="day_of_week" value="2" min="0" max="6"></div>
<div class="form-group"><label>International Transaction</label><select id="is_international"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
</div>
<div style="display:flex;gap:12px;margin-top:8px;"><button class="btn btn-primary" onclick="predictFraud()">Score Transaction</button><button class="btn btn-secondary" onclick="loadSample()">Load Random Sample</button></div>
</div>
<div class="result-box" id="result"><h3 id="result-title"></h3><div id="result-body"></div></div>
<div class="card" style="margin-top:20px;"><div class="card-title">How It Works</div><p style="font-size:0.88rem;color:var(--text-muted);line-height:1.6;">XGBoost trained on 120,000 transactions with 5.2% fraud rate. AUC: 0.90.</p></div>"""

    js = """function val(id){return document.getElementById(id).value;}
function predictFraud(){
var d={amount:val('amount'),txn_type:val('txn_type'),channel:val('channel'),merchant:val('merchant'),hour_of_day:val('hour_of_day'),day_of_week:val('day_of_week'),is_international:val('is_international')};
fetch('/api/fraud/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(function(r){return r.json();}).then(function(res){
var box=document.getElementById('result');var cls='approve';
if(res.fraud_probability>40)cls='danger';else if(res.fraud_probability>15)cls='warn';
box.className='result-box show '+cls;
var color=res.fraud_probability>40?'#e53935':res.fraud_probability>15?'#fb8c00':'#43a047';
document.getElementById('result-title').innerHTML='Fraud Score: '+res.fraud_probability+'% — '+res.prediction;
var ih='';
for(var i=0;i<res.risk_indicators.length;i++){ih+='<div class="info-item"><span class="key">•</span><span class="val">'+res.risk_indicators[i]+'</span></div>';}
document.getElementById('result-body').innerHTML='<p style="font-size:1rem;font-weight:600;margin-bottom:8px;">'+res.action+'</p><div class="prob-bar"><div class="prob-fill" style="width:'+res.fraud_probability+'%;background:'+color+';"></div></div><div class="divider"></div><p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:6px;">Risk Indicators:</p>'+ih;
}).catch(function(err){document.getElementById('result').className='result-box show danger';document.getElementById('result-title').innerHTML='Error';document.getElementById('result-body').innerHTML='<p>'+err.message+'</p>';});
}
function loadSample(){fetch('/api/fraud/sample').then(function(r){return r.json();}).then(function(s){document.getElementById('amount').value=s.amount;document.getElementById('txn_type').value=s.txn_type;document.getElementById('channel').value=s.channel;document.getElementById('merchant').value=s.merchant;document.getElementById('hour_of_day').value=s.hour_of_day;document.getElementById('day_of_week').value=s.day_of_week;document.getElementById('is_international').value=s.is_international;});}"""
    return page("Fraud Detection", "Real-time transaction fraud scoring using XGBoost", content, js)


@app.route('/api/fraud/predict', methods=['POST'])
def fraud_predict():
    _load_models()
    data = request.json
    input_data = pd.DataFrame([{
        'amount': float(data['amount']), 'amount_log': np.log1p(float(data['amount'])),
        'txn_type': data['txn_type'], 'channel': data['channel'], 'merchant': data['merchant'],
        'hour_of_day': int(data['hour_of_day']), 'day_of_week': int(data['day_of_week']),
        'is_international': int(data.get('is_international', 0)),
        'is_night_txn': int(1 if (int(data['hour_of_day']) < 6 or int(data['hour_of_day']) > 22) else 0),
        'is_weekend': int(1 if int(data['day_of_week']) >= 5 else 0),
        'is_high_amount': int(1 if float(data['amount']) > 50000 else 0),
    }])
    input_encoded = encode_input(input_data, _fraud_features, 'fraud', _encoders)[_fraud_features]
    proba = _fraud_model.predict_proba(input_encoded)[0]
    fp = float(proba[1])
    if fp > 0.7: action = 'BLOCK TRANSACTION — High fraud probability'
    elif fp > 0.4: action = 'STEP-UP AUTH — Require OTP/biometric verification'
    elif fp > 0.15: action = 'MONITOR — Flag for review, allow transaction'
    else: action = 'APPROVE — Low risk transaction'
    return jsonify({
        'fraud_probability': round(fp * 100, 2), 'prediction': 'FRAUD' if fp > 0.5 else 'LEGITIMATE',
        'action': action,
        'risk_indicators': [
            'Night transaction (' + ('Yes' if int(data['hour_of_day']) < 6 or int(data['hour_of_day']) > 22 else 'No') + ')',
            'International (' + ('Yes' if int(data.get('is_international', 0)) else 'No') + ')',
            'High amount >Rs50k (' + ('Yes' if float(data['amount']) > 50000 else 'No') + ')',
            'Weekend (' + ('Yes' if int(data['day_of_week']) >= 5 else 'No') + ')',
        ]
    })


@app.route('/api/fraud/sample')
def fraud_sample():
    _load_data()
    s = txns_df.sample(1).iloc[0]
    return jsonify({'amount': float(s['amount']), 'txn_type': str(s['txn_type']), 'channel': str(s['channel']),
        'merchant': str(s['merchant']), 'hour_of_day': int(s['hour_of_day']), 'day_of_week': int(s['day_of_week']),
        'is_international': int(s['is_international']), 'actual_fraud': int(s['is_fraud'])})


@app.route('/kyc-aml')
def kyc_page():
    content = """<div class="card"><div class="card-title">Entity Information</div>
<div class="form-grid">
<div class="form-group"><label>Entity Type</label><select id="entity_type"><option value="Individual" selected>Individual</option><option value="Corporate">Corporate</option><option value="Partnership">Partnership</option><option value="Trust">Trust</option></select></div>
<div class="form-group"><label>Country</label><select id="country"><option value="India" selected>India</option><option value="USA">USA</option><option value="UK">UK</option><option value="Singapore">Singapore</option><option value="UAE">UAE</option><option value="Switzerland">Switzerland</option><option value="Hong Kong">Hong Kong</option><option value="Germany">Germany</option><option value="Cyprus">Cyprus (High Risk)</option><option value="BVI">BVI (High Risk)</option><option value="Panama">Panama (High Risk)</option><option value="Seychelles">Seychelles (High Risk)</option><option value="Mauritius">Mauritius (High Risk)</option><option value="Cayman Islands">Cayman Islands (High Risk)</option><option value="China">China</option></select></div>
<div class="form-group"><label>Document Type</label><select id="doc_type"><option value="PAN" selected>PAN</option><option value="Passport">Passport</option><option value="Aadhaar">Aadhaar</option><option value="Driving License">Driving License</option><option value="Voter ID">Voter ID</option></select></div>
<div class="form-group"><label>Document Verified</label><select id="doc_verified"><option value="1" selected>Yes</option><option value="0">No</option></select></div>
<div class="form-group"><label>Number of Bank Accounts</label><input type="number" id="num_bank_accounts" value="2" min="1"></div>
<div class="form-group"><label>Monthly Transaction Volume (₹)</label><input type="number" id="txn_volume_monthly" value="500000" step="50000"></div>
<div class="form-group"><label>PEP Link</label><select id="has_pep_link"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
<div class="form-group"><label>Sanction Link</label><select id="has_sanction_link"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
<div class="form-group"><label>Adverse Media Hits</label><input type="number" id="adverse_media_hits" value="0" min="0" max="10"></div>
<div class="form-group"><label>Years in Business</label><input type="number" id="years_in_business" value="10" min="1"></div>
<div class="form-group"><label>Ownership Transparency</label><select id="ownership_transparency"><option value="High" selected>High</option><option value="Medium">Medium</option><option value="Low">Low</option></select></div>
</div>
<div style="display:flex;gap:12px;margin-top:8px;"><button class="btn btn-primary" onclick="predictKYC()">Assess AML Risk</button><button class="btn btn-secondary" onclick="loadSample()">Load Random Sample</button></div>
</div>
<div class="result-box" id="result"><h3 id="result-title"></h3><div id="result-body"></div></div>
<div class="card" style="margin-top:20px;"><div class="card-title">How It Works</div><p style="font-size:0.88rem;color:var(--text-muted);line-height:1.6;">Random Forest + rule-based risk scoring. Evaluates 13 features including sanction/PEP links and jurisdiction risk.</p></div>"""

    js = """function val(id){return document.getElementById(id).value;}
function predictKYC(){
var d={entity_type:val('entity_type'),country:val('country'),doc_type:val('doc_type'),doc_verified:val('doc_verified'),num_bank_accounts:val('num_bank_accounts'),txn_volume_monthly:val('txn_volume_monthly'),has_pep_link:val('has_pep_link'),has_sanction_link:val('has_sanction_link'),adverse_media_hits:val('adverse_media_hits'),years_in_business:val('years_in_business'),ownership_transparency:val('ownership_transparency')};
fetch('/api/kyc/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(function(r){return r.json();}).then(function(res){
var box=document.getElementById('result');var cls='approve';
if(res.risk_category==='Critical'||res.risk_category==='High')cls='danger';else if(res.risk_category==='Medium')cls='warn';
box.className='result-box show '+cls;
var color=res.risk_score>40?'#e53935':res.risk_score>20?'#fb8c00':'#43a047';
document.getElementById('result-title').innerHTML='Risk Score: '+res.risk_score+'/100 — Category: '+res.risk_category;
var fh='';
for(var i=0;i<res.risk_factors.length;i++){fh+='<div class="info-item"><span class="key">•</span><span class="val">'+res.risk_factors[i]+'</span></div>';}
document.getElementById('result-body').innerHTML='<p style="font-size:1rem;font-weight:600;margin-bottom:8px;">'+res.action+'</p><div class="prob-bar"><div class="prob-fill" style="width:'+res.risk_score+'%;background:'+color+';"></div></div><div class="divider"></div><div class="info-grid"><div class="info-item"><span class="key">Alert Probability</span><span class="val">'+res.alert_probability+'%</span></div><div class="info-item"><span class="key">Alert Status</span><span class="val">'+res.alert+'</span></div></div><div class="divider"></div><p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:6px;">Risk Factors:</p>'+fh;
}).catch(function(err){document.getElementById('result').className='result-box show danger';document.getElementById('result-title').innerHTML='Error';document.getElementById('result-body').innerHTML='<p>'+err.message+'</p>';});
}
function loadSample(){fetch('/api/kyc/sample').then(function(r){return r.json();}).then(function(s){document.getElementById('entity_type').value=s.entity_type;document.getElementById('country').value=s.country;document.getElementById('doc_type').value=s.doc_type;document.getElementById('doc_verified').value=s.doc_verified;document.getElementById('num_bank_accounts').value=s.num_bank_accounts;document.getElementById('txn_volume_monthly').value=s.txn_volume_monthly;document.getElementById('has_pep_link').value=s.has_pep_link;document.getElementById('has_sanction_link').value=s.has_sanction_link;document.getElementById('adverse_media_hits').value=s.adverse_media_hits;document.getElementById('years_in_business').value=s.years_in_business;document.getElementById('ownership_transparency').value=s.ownership_transparency;});}"""
    return page("KYC / AML Risk Assessment", "AI-powered compliance risk scoring", content, js)


@app.route('/api/kyc/predict', methods=['POST'])
def kyc_predict():
    _load_models()
    data = request.json
    risk_countries = {'Cyprus', 'BVI', 'Panama', 'Seychelles', 'Mauritius', 'Cayman Islands'}
    input_data = pd.DataFrame([{
        'entity_type': data['entity_type'], 'country': data['country'], 'doc_type': data['doc_type'],
        'doc_verified': int(data['doc_verified']), 'num_bank_accounts': int(data['num_bank_accounts']),
        'txn_volume_monthly': float(data['txn_volume_monthly']),
        'has_pep_link': int(data.get('has_pep_link', 0)), 'has_sanction_link': int(data.get('has_sanction_link', 0)),
        'adverse_media_hits': int(data.get('adverse_media_hits', 0)), 'years_in_business': int(data['years_in_business']),
        'ownership_transparency': data['ownership_transparency'],
        'txn_volume_log': np.log1p(float(data['txn_volume_monthly'])),
        'high_risk_country': int(1 if data['country'] in risk_countries else 0),
    }])
    input_encoded = encode_input(input_data, _kyc_features, 'kyc', _encoders)[_kyc_features]
    proba = _kyc_model.predict_proba(input_encoded)[0]
    ap = float(proba[1])
    rs = min(100, int(
        int(data.get('has_sanction_link', 0)) * 40 + int(data.get('has_pep_link', 0)) * 15 +
        int(data.get('adverse_media_hits', 0)) * 8 + (1 if data['country'] in risk_countries else 0) * 12 +
        (1 if data['ownership_transparency'] == 'Low' else 0) * 6 + (1 if int(data['doc_verified']) == 0 else 0) * 5 +
        max(0, np.log1p(float(data['txn_volume_monthly'])) - 12) * 0.3))
    if rs <= 20: cat = 'Low'
    elif rs <= 40: cat = 'Medium'
    elif rs <= 60: cat = 'High'
    else: cat = 'Critical'
    if cat in ('High', 'Critical') or int(data.get('has_sanction_link', 0)):
        action = 'ESCALATE — Enhanced due diligence required, freeze onboarding'
    elif cat == 'Medium': action = 'REVIEW — Additional documentation requested'
    else: action = 'CLEAR — Standard onboarding approved'
    return jsonify({
        'risk_score': rs, 'risk_category': cat, 'alert_probability': round(ap * 100, 2),
        'alert': 'YES' if ap > 0.5 else 'NO', 'action': action,
        'risk_factors': [
            'Sanction link: ' + ('YES' if int(data.get('has_sanction_link', 0)) else 'No'),
            'PEP link: ' + ('YES' if int(data.get('has_pep_link', 0)) else 'No'),
            'High-risk country: ' + ('YES' if data['country'] in risk_countries else 'No'),
            'Adverse media hits: ' + str(int(data.get('adverse_media_hits', 0))),
            'Ownership transparency: ' + data['ownership_transparency'],
            'Document verified: ' + ('No' if int(data['doc_verified']) == 0 else 'Yes'),
        ]
    })


@app.route('/api/kyc/sample')
def kyc_sample():
    _load_data()
    s = kyc_df.sample(1).iloc[0]
    return jsonify({'entity_type': str(s['entity_type']), 'country': str(s['country']), 'doc_type': str(s['doc_type']),
        'doc_verified': int(s['doc_verified']), 'num_bank_accounts': int(s['num_bank_accounts']),
        'txn_volume_monthly': float(s['txn_volume_monthly']), 'has_pep_link': int(s['has_pep_link']),
        'has_sanction_link': int(s['has_sanction_link']), 'adverse_media_hits': int(s['adverse_media_hits']),
        'years_in_business': int(s['years_in_business']), 'ownership_transparency': str(s['ownership_transparency']),
        'actual_alert': int(s['alert_flag'])})


@app.route('/portfolio-advisor')
def portfolio_page():
    content = """<div class="card"><div class="card-title">Investment Profile</div>
<div class="form-grid">
<div class="form-group"><label>Risk Profile</label><select id="risk_profile"><option value="Conservative">Conservative</option><option value="Moderate" selected>Moderate</option><option value="Aggressive">Aggressive</option><option value="Very Aggressive">Very Aggressive</option></select></div>
<div class="form-group"><label>Investment Horizon (years)</label><select id="investment_horizon"><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="5" selected>5</option><option value="7">7</option><option value="10">10</option><option value="15">15</option><option value="20">20</option></select></div>
<div class="form-group"><label>Investment Capital (₹)</label><input type="number" id="capital" value="1000000" step="100000"></div>
</div>
<button class="btn btn-primary" onclick="optimize()">Generate Optimized Portfolio</button>
</div>
<div class="result-box" id="result"><h3 id="result-title"></h3><div id="result-body"></div></div>
<div class="card" style="margin-top:20px;"><div class="card-title">How It Works</div><p style="font-size:0.88rem;color:var(--text-muted);line-height:1.6;">Modern Portfolio Theory — balancing return against volatility to maximize Sharpe ratio.</p></div>"""

    js = """function val(id){return document.getElementById(id).value;}
var AC={'Equity_LargeCap':'#1a237e','Equity_MidCap':'#3949ab','Equity_SmallCap':'#5c6bc0','Debt_Govt':'#00897b','Debt_Corporate':'#00bfa5','Gold':'#fdd835','International':'#fb8c00','REIT':'#8e24aa','Commodities':'#e53935','Cash':'#9e9e9e'};
var AL={'Equity_LargeCap':'Equity (Large Cap)','Equity_MidCap':'Equity (Mid Cap)','Equity_SmallCap':'Equity (Small Cap)','Debt_Govt':'Debt (Govt)','Debt_Corporate':'Debt (Corporate)','Gold':'Gold','International':'International Equity','REIT':'REIT','Commodities':'Commodities','Cash':'Cash'};
function optimize(){
var d={risk_profile:val('risk_profile'),investment_horizon:val('investment_horizon'),capital:val('capital')};
fetch('/api/portfolio/optimize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(function(r){return r.json();}).then(function(res){
var box=document.getElementById('result');box.className='result-box show approve';
document.getElementById('result-title').innerHTML='Optimized Portfolio — '+res.risk_profile+' | '+res.investment_horizon+'Y Horizon';
var ab='<div class="allocation-bar">';
var keys=Object.keys(res.allocation);
for(var i=0;i<keys.length;i++){var a=keys[i];var p=res.allocation[a];if(p>0.01){ab+='<div class="alloc-segment" style="width:'+(p*100)+'%;background:'+AC[a]+'" title="'+AL[a]+': '+(p*100).toFixed(1)+'%">'+(p>0.05?(p*100).toFixed(0)+'%':'')+'</div>';}}
ab+='</div>';
var rows='';
for(var j=0;j<keys.length;j++){var a2=keys[j];var p2=res.allocation[a2];if(p2>0){rows+='<div class="info-item"><span class="key"><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:'+AC[a2]+';margin-right:6px;"></span>'+AL[a2]+'</span><span class="val">'+(p2*100).toFixed(1)+'% — Rs '+res.allocation_amounts[a2].toLocaleString('en-IN')+'</span></div>';}}
document.getElementById('result-body').innerHTML='<div class="info-grid"><div class="info-item"><span class="key">Expected Return (Annual)</span><span class="val" style="color:#43a047;">'+res.expected_return+'%</span></div><div class="info-item"><span class="key">Expected Volatility</span><span class="val" style="color:#fb8c00;">'+res.expected_volatility+'%</span></div><div class="info-item"><span class="key">Sharpe Ratio</span><span class="val">'+res.sharpe_ratio+'</span></div><div class="info-item"><span class="key">Capital</span><span class="val">Rs '+parseInt(res.capital).toLocaleString('en-IN')+'</span></div></div><div class="divider"></div><p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:6px;">Asset Allocation:</p>'+ab+'<div class="divider"></div>'+rows;
}).catch(function(err){document.getElementById('result').className='result-box show danger';document.getElementById('result-title').innerHTML='Error';document.getElementById('result-body').innerHTML='<p>'+err.message+'</p>';});
}"""
    return page("Investment Portfolio Advisor", "AI-driven asset allocation optimization", content, js)


@app.route('/api/portfolio/optimize', methods=['POST'])
def portfolio_optimize():
    _load_models()
    data = request.json
    risk_profile = data['risk_profile']
    horizon = int(data.get('investment_horizon', 5))
    capital = float(data.get('capital', 1000000))
    allocations = {
        'Conservative': {'Equity_LargeCap':0.15,'Equity_MidCap':0.05,'Equity_SmallCap':0.0,'Debt_Govt':0.30,'Debt_Corporate':0.20,'Gold':0.10,'International':0.05,'REIT':0.05,'Commodities':0.03,'Cash':0.07},
        'Moderate': {'Equity_LargeCap':0.25,'Equity_MidCap':0.10,'Equity_SmallCap':0.05,'Debt_Govt':0.20,'Debt_Corporate':0.15,'Gold':0.08,'International':0.07,'REIT':0.05,'Commodities':0.03,'Cash':0.02},
        'Aggressive': {'Equity_LargeCap':0.30,'Equity_MidCap':0.15,'Equity_SmallCap':0.10,'Debt_Govt':0.10,'Debt_Corporate':0.10,'Gold':0.05,'International':0.10,'REIT':0.05,'Commodities':0.04,'Cash':0.01},
        'Very Aggressive': {'Equity_LargeCap':0.25,'Equity_MidCap':0.20,'Equity_SmallCap':0.15,'Debt_Govt':0.05,'Debt_Corporate':0.05,'Gold':0.05,'International':0.15,'REIT':0.05,'Commodities':0.04,'Cash':0.01},
    }
    alloc = allocations[risk_profile].copy()
    if horizon > 10:
        for eq in ['Equity_LargeCap','Equity_MidCap','Equity_SmallCap']:
            alloc[eq] = round(alloc[eq] * 1.15, 4)
        total = sum(alloc.values())
        alloc = {k: round(v/total, 4) for k, v in alloc.items()}
    ret = sum(alloc[a] * _asset_stats[a][0] for a in alloc)
    vol = np.sqrt(sum((alloc[a] * _asset_stats[a][1])**2 for a in alloc) +
                  2 * sum(alloc[a] * alloc[b] * _asset_stats[a][1] * _asset_stats[b][1] * 0.3
                          for i, a in enumerate(alloc) for b in list(alloc)[i+1:]))
    sharpe = (ret - 0.035) / vol if vol > 0 else 0
    return jsonify({
        'allocation': alloc, 'allocation_amounts': {k: round(v * capital) for k, v in alloc.items()},
        'expected_return': round(ret * 100, 2), 'expected_volatility': round(vol * 100, 2),
        'sharpe_ratio': round(sharpe, 3), 'risk_profile': risk_profile,
        'investment_horizon': horizon, 'capital': capital,
    })


@app.route('/churn')
def churn_page():
    content = """<div class="card"><div class="card-title">Customer Profile</div>
<div class="form-grid">
<div class="form-group"><label>Age</label><input type="number" id="age" value="42"></div>
<div class="form-group"><label>Annual Income (₹)</label><input type="number" id="annual_income" value="850000" step="50000"></div>
<div class="form-group"><label>Months with Bank</label><input type="number" id="months_on_book" value="60"></div>
<div class="form-group"><label>Number of Products</label><input type="number" id="num_products" value="3" min="1" max="5"></div>
<div class="form-group"><label>Avg Monthly Balance (₹)</label><input type="number" id="avg_monthly_balance" value="75000" step="5000"></div>
<div class="form-group"><label>Digital Engagement Score (0-100)</label><input type="number" id="digital_engagement_score" value="45" min="0" max="100"></div>
<div class="form-group"><label>Complaints (Last Year)</label><input type="number" id="complaints_last_year" value="2" min="0"></div>
<div class="form-group"><label>Branch Visits</label><input type="number" id="num_branch_visits" value="5" min="0"></div>
<div class="form-group"><label>Credit Card Usage Score (0-100)</label><input type="number" id="credit_card_usage" value="30" min="0" max="100"></div>
<div class="form-group"><label>Gender</label><select id="gender"><option value="M" selected>Male</option><option value="F">Female</option></select></div>
<div class="form-group"><label>Segment</label><select id="segment"><option value="Retail" selected>Retail</option><option value="Mass Affluent">Mass Affluent</option><option value="HNWI">HNWI</option><option value="Corporate">Corporate</option><option value="SME">SME</option></select></div>
<div class="form-group"><label>Employment Type</label><select id="employment_type"><option value="Salaried" selected>Salaried</option><option value="Self-Employed">Self-Employed</option><option value="Business Owner">Business Owner</option><option value="Retired">Retired</option></select></div>
</div>
<div style="display:flex;gap:12px;margin-top:8px;"><button class="btn btn-primary" onclick="predictChurn()">Predict Churn Risk</button><button class="btn btn-secondary" onclick="loadSample()">Load Random Sample</button></div>
</div>
<div class="result-box" id="result"><h3 id="result-title"></h3><div id="result-body"></div></div>
<div class="card" style="margin-top:20px;"><div class="card-title">How It Works</div><p style="font-size:0.88rem;color:var(--text-muted);line-height:1.6;">XGBoost trained on 8,000 customer records with 3.2% churn rate.</p></div>"""

    js = """function val(id){return document.getElementById(id).value;}
function predictChurn(){
var d={age:val('age'),annual_income:val('annual_income'),months_on_book:val('months_on_book'),num_products:val('num_products'),avg_monthly_balance:val('avg_monthly_balance'),digital_engagement_score:val('digital_engagement_score'),complaints_last_year:val('complaints_last_year'),num_branch_visits:val('num_branch_visits'),credit_card_usage:val('credit_card_usage'),gender:val('gender'),segment:val('segment'),employment_type:val('employment_type')};
fetch('/api/churn/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(function(r){return r.json();}).then(function(res){
var box=document.getElementById('result');var cls='approve';
if(res.churn_probability>25)cls='danger';else if(res.churn_probability>10)cls='warn';
box.className='result-box show '+cls;
var color=res.churn_probability>25?'#e53935':res.churn_probability>10?'#fb8c00':'#43a047';
document.getElementById('result-title').innerHTML='Churn Probability: '+res.churn_probability+'% — '+res.prediction;
var fh='';
for(var i=0;i<res.risk_factors.length;i++){fh+='<div class="info-item"><span class="key">•</span><span class="val">'+res.risk_factors[i]+'</span></div>';}
document.getElementById('result-body').innerHTML='<p style="font-size:1rem;font-weight:600;margin-bottom:8px;">'+res.action+'</p><div class="prob-bar"><div class="prob-fill" style="width:'+res.churn_probability+'%;background:'+color+';"></div></div><div class="divider"></div><p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:6px;">Risk Factors:</p>'+fh;
}).catch(function(err){document.getElementById('result').className='result-box show danger';document.getElementById('result-title').innerHTML='Error';document.getElementById('result-body').innerHTML='<p>'+err.message+'</p>';});
}
function loadSample(){fetch('/api/churn/sample').then(function(r){return r.json();}).then(function(s){document.getElementById('age').value=s.age;document.getElementById('annual_income').value=s.annual_income;document.getElementById('months_on_book').value=s.months_on_book;document.getElementById('num_products').value=s.num_products;document.getElementById('avg_monthly_balance').value=s.avg_monthly_balance;document.getElementById('digital_engagement_score').value=s.digital_engagement_score;document.getElementById('complaints_last_year').value=s.complaints_last_year;document.getElementById('num_branch_visits').value=s.num_branch_visits;document.getElementById('credit_card_usage').value=s.credit_card_usage;document.getElementById('gender').value=s.gender;document.getElementById('segment').value=s.segment;document.getElementById('employment_type').value=s.employment_type;});}"""
    return page("Customer Churn Prediction", "AI-powered customer retention scoring", content, js)


@app.route('/api/churn/predict', methods=['POST'])
def churn_predict():
    _load_models()
    data = request.json
    input_data = pd.DataFrame([{
        'age': int(data['age']), 'annual_income': float(data['annual_income']),
        'months_on_book': int(data['months_on_book']), 'num_products': int(data['num_products']),
        'income_log': np.log1p(float(data['annual_income'])),
        'avg_monthly_balance': float(data['avg_monthly_balance']),
        'digital_engagement_score': int(data['digital_engagement_score']),
        'complaints_last_year': int(data['complaints_last_year']),
        'num_branch_visits': int(data['num_branch_visits']),
        'credit_card_usage': int(data['credit_card_usage']),
        'gender': data.get('gender', 'M'), 'segment': data.get('segment', 'Retail'),
        'employment_type': data.get('employment_type', 'Salaried'),
    }])
    input_encoded = encode_input(input_data, _churn_features, 'churn', _encoders)[_churn_features]
    proba = _churn_model.predict_proba(input_encoded)[0]
    cp = float(proba[1])
    if cp > 0.5: action = 'RETENTION CAMPAIGN — High churn risk, immediate intervention needed'
    elif cp > 0.25: action = 'PROACTIVE OUTREACH — Offer personalized benefits or fee waivers'
    elif cp > 0.10: action = 'MONITOR — Include in watchlist, increase engagement touchpoints'
    else: action = 'LOW RISK — No action needed'
    return jsonify({
        'churn_probability': round(cp * 100, 2), 'prediction': 'CHURN' if cp > 0.5 else 'RETAIN',
        'action': action,
        'risk_factors': [
            'Digital engagement: ' + str(data['digital_engagement_score']) + '/100',
            'Complaints (last year): ' + str(data['complaints_last_year']),
            'Products held: ' + str(data['num_products']),
            'Branch visits: ' + str(data['num_branch_visits']),
        ]
    })


@app.route('/api/churn/sample')
def churn_sample():
    _load_data()
    s = churn_df.sample(1).iloc[0]
    return jsonify({'age': int(s['age']), 'annual_income': int(s['annual_income']),
        'months_on_book': int(s['months_on_book']), 'num_products': int(s['num_products']),
        'avg_monthly_balance': int(s['avg_monthly_balance']),
        'digital_engagement_score': int(s['digital_engagement_score']),
        'complaints_last_year': int(s['complaints_last_year']),
        'num_branch_visits': int(s['num_branch_visits']),
        'credit_card_usage': int(s['credit_card_usage']),
        'gender': str(s['gender']), 'segment': str(s['segment']),
        'employment_type': str(s['employment_type']), 'actual_churn': int(s['churn'])})


@app.route('/models')
def models_page():
    _load_models()
    m = _model_metrics
    rows = ""
    for key, val in m.items():
        algo = "Random Forest" if key == "kyc_aml" else "XGBoost"
        domain = {'credit_risk':'Retail Banking','fraud_detection':'Retail / Payments','kyc_aml':'Commercial Banking','churn':'Retail Banking'}[key]
        name = key.replace('_', ' ').title()
        rate_label = {'credit_risk':'Default Rate','fraud_detection':'Fraud Rate','kyc_aml':'Alert Rate','churn':'Churn Rate'}[key]
        rate_key = {'credit_risk':'default_rate','fraud_detection':'fraud_rate','kyc_aml':'alert_rate','churn':'churn_rate'}[key]
        rate_val = val[rate_key] * 100
        cm = val['confusion_matrix']
        fi_html = ""
        for fi in val['feature_importance'][:8]:
            pct = round(fi['importance'] * 100, 1)
            fi_html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;"><span style="font-size:0.82rem;width:180px;">' + fi['feature'] + '</span><div style="flex:1;height:16px;background:#f0f0f0;border-radius:8px;overflow:hidden;"><div style="height:100%;width:' + str(pct) + '%;background:#1a237e;border-radius:8px;"></div></div><span style="font-size:0.78rem;font-weight:600;width:40px;text-align:right;">' + str(pct) + '%</span></div>'
        rows += '<div class="card"><div class="card-title">' + name + '<span style="float:right;font-size:0.8rem;font-weight:400;color:var(--text-muted);">AUC: ' + str(val['auc']) + ' &middot; F1: ' + str(val['f1']) + '</span></div>'
        rows += '<div class="info-grid" style="grid-template-columns:repeat(4,1fr);">'
        rows += '<div class="info-item"><span class="key">Algorithm</span><span class="val">' + algo + '</span></div>'
        rows += '<div class="info-item"><span class="key">Accuracy</span><span class="val">' + str(val['accuracy']) + '</span></div>'
        rows += '<div class="info-item"><span class="key">Precision</span><span class="val">' + str(val['precision']) + '</span></div>'
        rows += '<div class="info-item"><span class="key">Recall</span><span class="val">' + str(val['recall']) + '</span></div>'
        rows += '<div class="info-item"><span class="key">Train Samples</span><span class="val">' + format(val['train_samples'], ',') + '</span></div>'
        rows += '<div class="info-item"><span class="key">Test Samples</span><span class="val">' + format(val['test_samples'], ',') + '</span></div>'
        rows += '<div class="info-item"><span class="key">Features</span><span class="val">' + str(val['feature_count']) + '</span></div>'
        rows += '<div class="info-item"><span class="key">' + rate_label + '</span><span class="val">' + str(round(rate_val, 1)) + '%</span></div>'
        rows += '</div><div class="divider"></div>'
        rows += '<p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:6px;">Confusion Matrix:</p>'
        rows += '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;max-width:300px;margin-bottom:16px;">'
        rows += '<div style="background:#f5f6fa;padding:8px;text-align:center;border-radius:6px;font-size:0.78rem;color:var(--text-muted);">Actual \\ Pred</div>'
        rows += '<div style="background:#e8f5e9;padding:8px;text-align:center;border-radius:6px;font-size:0.78rem;">Neg (0)</div>'
        rows += '<div style="background:#ffebee;padding:8px;text-align:center;border-radius:6px;font-size:0.78rem;">Pos (1)</div>'
        rows += '<div style="background:#e8f5e9;padding:8px;text-align:center;border-radius:6px;font-size:0.78rem;">Neg (0)</div>'
        rows += '<div style="background:#e8f5e9;padding:10px;text-align:center;border-radius:6px;font-weight:700;">' + str(cm[0][0]) + '</div>'
        rows += '<div style="background:#fff3e0;padding:10px;text-align:center;border-radius:6px;font-weight:700;">' + str(cm[0][1]) + '</div>'
        rows += '<div style="background:#ffebee;padding:8px;text-align:center;border-radius:6px;font-size:0.78rem;">Pos (1)</div>'
        rows += '<div style="background:#fff3e0;padding:10px;text-align:center;border-radius:6px;font-weight:700;">' + str(cm[1][0]) + '</div>'
        rows += '<div style="background:#e8f5e9;padding:10px;text-align:center;border-radius:6px;font-weight:700;">' + str(cm[1][1]) + '</div>'
        rows += '</div><p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:6px;">Top Features by Importance:</p>'
        rows += '<div style="max-width:500px;">' + fi_html + '</div></div>'
    content = rows + '<div class="footer-note">All models trained on synthetic data. For production, retrain on real banking data with MLOps pipelines.</div>'
    return page("Model Performance", "Detailed evaluation metrics for all banking AI models", content, "")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
