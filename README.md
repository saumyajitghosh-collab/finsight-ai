# FinSight AI — Banking Intelligence Platform

A cross-functional AI application built for the banking sector, spanning **Retail Banking**, **Commercial/Institutional Banking**, and **Investment Banking**. The platform uses trained machine learning models on realistic synthetic banking data to deliver five distinct AI capabilities through an interactive web dashboard.

## Modules

| Module | Banking Domain | ML Model | Key Use Case |
|--------|----------------|----------|--------------|
| Credit Risk Scoring | Retail Banking | XGBoost | Predicts loan default probability, risk grade (A–E), lending recommendation |
| Fraud Detection | Retail / Payments | XGBoost | Real-time transaction fraud scoring with block/step-up-auth/monitor/approve actions |
| KYC / AML Risk | Commercial Banking | Random Forest + Rules | Compliance risk scoring (0–100), Low/Medium/High/Critical, EDD escalation |
| Portfolio Advisor | Investment Banking | Modern Portfolio Theory | Optimized asset allocation across 10 asset classes, Sharpe ratio maximization |
| Churn Prediction | Retail Banking | XGBoost | Customer churn risk scoring with retention campaign recommendations |

## Quick Start (Local)

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploy to Render (Free, Public URL)

1. Create a GitHub repo and upload all files from this folder
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click "Create Web Service"
6. Wait 3–5 minutes for build & deploy
7. Your app is live at `https://your-app-name.onrender.com`

See `DEPLOYMENT_GUIDE.md` for detailed step-by-step instructions with screenshots.

## Project Structure

```
banking_ai_app/
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── Procfile                # Render/gunicorn start command
├── Dockerfile              # Container deployment option
├── runtime.txt             # Python version
├── .gitignore
├── data/                   # Synthetic banking datasets (CSV)
├── models/                 # Trained ML models (joblib)
├── templates/              # HTML templates (Jinja2)
├── build_data_and_models.py  # Data generation script
└── train_models.py           # Model training script
```

## Disclaimer

All data is synthetic. For production deployment, models must be retrained on real data with proper regulatory compliance, model validation, and MLOps monitoring.
