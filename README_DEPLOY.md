# Clinical Coordinator Dashboard

Oura Ring Study Data Monitoring Dashboard

## 🚀 Quick Deploy to Render

### Step 1: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `oura-clinical-dashboard`
3. Keep it **Private** (for patient data security)
4. Click "Create repository"

### Step 2: Push Code to GitHub

Run these commands in your terminal:

```bash
cd "/Users/albinakrasykova/Desktop/oura study"

# Initialize git (if not already)
git init

# Add deployment files
cp .gitignore_deploy .gitignore

# Add files (NOT real patient data)
git add app.py templates/ requirements_deploy.txt render.yaml demo_data.xlsx README_DEPLOY.md

# Commit
git commit -m "Clinical dashboard for deployment"

# Add your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/oura-clinical-dashboard.git

# Push
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account
4. Select your `oura-clinical-dashboard` repo
5. Settings:
   - **Name**: `oura-clinical-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements_deploy.txt`
   - **Start Command**: `gunicorn app:app`
6. Click **"Create Web Service"**

### Step 4: Get Your Link! 🎉

After ~2 minutes, you'll get a link like:
```
https://oura-clinical-dashboard.onrender.com
```

Share this with your advisor!

## 📁 Files for Deployment

```
├── app.py                 # Flask application
├── templates/
│   └── dashboard.html     # Dashboard UI
├── requirements_deploy.txt # Python dependencies
├── render.yaml            # Render configuration
├── demo_data.xlsx         # Sample data (safe to share)
└── .gitignore            # Keeps secrets safe
```

## ⚠️ Security Notes

- **NEVER commit** `data.xlsx` (real patient data)
- **NEVER commit** `.env` files
- Use `demo_data.xlsx` for public demos
- Keep your GitHub repo **Private**

## 🔄 Updating the Dashboard

After making changes:
```bash
git add .
git commit -m "Update dashboard"
git push
```

Render will automatically redeploy!
