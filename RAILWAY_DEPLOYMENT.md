# 🚂 Railway Deployment Guide

This guide explains how to deploy the California House Price Prediction ML Pipeline to Railway.

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be pushed to GitHub
2. **Railway Account**: Sign up at [railway.app](https://railway.app)
3. **Trained Models**: The app will automatically train models during deployment

## 🚀 Deployment Steps

### 1. Connect GitHub to Railway

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository: `End-to-END-ML-Pipieline`

### 2. Configure Deployment

Railway will automatically detect the configuration from these files:
- `Procfile` - Defines how to start the application
- `requirements-railway.txt` - Python dependencies
- `railway.json` - Railway-specific configuration

### 3. Environment Variables (Optional)

If needed, you can set environment variables in Railway dashboard:
- `PORT` - Automatically set by Railway
- `PYTHONPATH` - Set to `.` if needed

### 4. Deploy

1. Click **"Deploy Now"**
2. Railway will:
   - Install dependencies from `requirements-railway.txt`
   - Run the startup script `railway_startup.py`
   - Train ML models automatically (if not present)
   - Start the Streamlit web application

### 5. Access Your App

Once deployed, Railway will provide:
- **Public URL**: `https://your-app-name.railway.app`
- **Deployment logs**: Monitor the training and startup process

## 🔧 Deployment Configuration

### Files Created for Railway:

1. **`Procfile`** - Starts the application
   ```
   web: python railway_startup.py
   ```

2. **`requirements-railway.txt`** - Core dependencies only
   ```
   streamlit>=1.28.0
   pandas>=2.0.0
   scikit-learn>=1.3.0
   xgboost>=1.7.0
   # ... other core deps
   ```

3. **`railway_startup.py`** - Handles model training and app startup
   - Checks if models exist
   - Trains models if missing
   - Starts Streamlit app

4. **`railway.json`** - Railway configuration
   ```json
   {
     "build": { "builder": "NIXPACKS" },
     "deploy": { "numReplicas": 1 }
   }
   ```

5. **`.railwayignore`** - Excludes unnecessary files

## ⏱️ Deployment Timeline

- **First deployment**: ~5-10 minutes (includes model training)
- **Subsequent deployments**: ~2-3 minutes (models cached)

## 🎯 Features After Deployment

✅ **Web Interface**: Interactive California house price prediction  
✅ **Multiple Models**: XGBoost, Random Forest, Linear Regression  
✅ **Location Dropdown**: Easy city selection instead of lat/long  
✅ **Realistic Predictions**: Proper dollar formatting ($400,000+ range)  
✅ **Feature Engineering**: Automatic 8→15 feature transformation  
✅ **Model Comparison**: Side-by-side predictions with performance metrics  

## 🛠️ Troubleshooting

### Common Issues:

1. **Build Timeout**: Model training takes too long
   - Solution: Push pre-trained models to GitHub

2. **Memory Issues**: Large model files
   - Solution: Use lighter model configurations

3. **Dependency Conflicts**: Package version mismatches
   - Solution: Use `requirements-railway.txt` with flexible versions

### Logs & Monitoring:

- View deployment logs in Railway dashboard
- Check for model training completion: "✅ Models trained successfully!"
- Monitor app startup: "🌐 Starting Streamlit application..."

## 🔄 Updates & Redeployment

To update your deployed app:
1. Push changes to your GitHub repository
2. Railway will automatically redeploy
3. Models are cached unless you change the training pipeline

## 📊 Expected Performance

- **Model Accuracy**: ~82% R² score (XGBoost)
- **Prediction Range**: $100,000 - $800,000 (realistic California prices)
- **Response Time**: <2 seconds for predictions
- **Uptime**: 99%+ (Railway SLA)

## 🌐 Public Access

Your deployed app will be accessible at:
```
https://your-project-name.railway.app
```

Share this URL to let others use your ML prediction app!

---

🎉 **Congratulations!** Your end-to-end ML pipeline is now live on Railway!