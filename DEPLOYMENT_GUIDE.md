# Deployment Guide — FinSight AI on Render

This guide walks you through hosting FinSight AI on the internet so anyone can access it via a public URL.

## Why Render?

- **Free tier** — no credit card required to start
- **Native Flask support** — reads your `Procfile` automatically
- **Handles XGBoost/scikit-learn** — all Python packages work out of the box
- **Auto-deploy from GitHub** — push code, it deploys
- **Public HTTPS URL** — `https://your-app.onrender.com`

## Prerequisites

- A [GitHub](https://github.com) account (free)
- The FinSight AI app files (from the zip)

---

## Step 1: Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `finsight-ai` (or any name you like)
3. Set to **Public** (so anyone can see/use it)
4. Check "Add a README file"
5. Click **Create repository**

## Step 2: Upload the App Files to GitHub

### Option A: Upload via Browser (Easiest)

1. Unzip `FinSight_AI_Banking_Platform.zip` on your computer
2. On your GitHub repo page, click **Add file → Upload files**
3. Drag and drop ALL files from the `banking_ai_app/` folder:
   - `app.py`
   - `Procfile`
   - `Dockerfile`
   - `requirements.txt`
   - `runtime.txt`
   - `.gitignore`
   - `README.md`
   - The entire `data/` folder
   - The entire `models/` folder
   - The entire `templates/` folder
4. Add a commit message: "Initial upload of FinSight AI app"
5. Click **Commit changes**

> **Note:** GitHub's web uploader has a 100-file limit per upload. If you hit it, upload in batches (data first, then models, then templates).

### Option B: Upload via Git Command Line (Faster for repeats)

```bash
# Unzip the file
unzip FinSight_AI_Banking_Platform.zip
cd banking_ai_app

# Initialize git and push
git init
git add .
git commit -m "Initial upload of FinSight AI app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/finsight-ai.git
git push -u origin main
```

## Step 3: Create a Render Account

1. Go to [render.com](https://render.com)
2. Click **Sign Up** → sign up with GitHub (connects your account automatically)
3. Authorize Render to access your GitHub repositories

## Step 4: Deploy the App

1. On the Render dashboard, click **New +** → **Web Service**
2. Find your `finsight-ai` repo → click **Connect**
3. Fill in the deployment settings:

   | Setting | Value |
   |---------|-------|
   | **Name** | `finsight-ai` (this becomes your URL) |
   | **Region** | Choose the closest (e.g., Singapore for India) |
   | **Branch** | `main` |
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app` |
   | **Instance Type** | **Free** |

4. Click **Create Web Service**

## Step 5: Wait for Deployment

- Render will now: install dependencies → build → start your app
- This takes **3–5 minutes** on first deploy
- Watch the logs for `==> Your service is live 🎉`
- Your app URL will be: `https://finsight-ai.onrender.com` (or whatever name you chose)

## Step 6: Verify It Works

Open your app URL in a browser. You should see:
- The dashboard with KPIs and charts
- All 5 modules working (Credit Risk, Fraud Detection, KYC/AML, Portfolio Advisor, Churn)
- The Model Performance page

Try clicking "Load Random Sample" on any module and then run a prediction.

---

## Important Notes

### Free Tier Limitations
- The app **sleeps after 15 minutes of inactivity** — the first request after sleep takes ~30 seconds to wake up
- 750 hours of free runtime per month (enough for 1 always-on service)
- 512 MB RAM (sufficient for this app)

### Keeping the App Awake
If you want it to respond instantly at all times:
- Upgrade to the **Starter plan** ($7/month) — no sleep, always responsive
- Or use a free uptime monitor like [UptimeRobot](https://uptimerobot.com) to ping your URL every 10 minutes (keeps it from sleeping)

### Updating the App
Any time you push new code to your GitHub repo's `main` branch, Render automatically redeploys. No manual action needed.

### Custom Domain
On a paid plan, you can add a custom domain like `finsightai.yourbank.com` in Render's settings.

---

## Alternative: Deploy with Docker

If you prefer Docker (works on Google Cloud Run, Railway, fly.io, etc.):

```bash
# Build the image
docker build -t finsight-ai .

# Run locally
docker run -p 5000:5000 finsight-ai

# Deploy to Google Cloud Run
gcloud run deploy finsight-ai --source . --region asia-south1
```

The included `Dockerfile` handles everything — Python 3.12, system dependencies for XGBoost, and gunicorn startup.
