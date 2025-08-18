#!/bin/bash

# Final Deployment Script for Agri-Credit Helper
set -e

echo "🚀 Starting deployment of Agri-Credit Helper..."

# --- Configuration ---
# Your Project ID is hardcoded here
PROJECT_ID="capitalone-468806"
REGION="asia-south1"
SERVICE_NAME="agri-credit-helper"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# --- Authenticate and Configure gcloud ---
echo "🔐 Setting Google Cloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# --- Build and Push the Docker Image ---
echo "🏗️ Building Docker image..."
docker build -t $IMAGE_NAME:latest .

echo "📤 Pushing image to Google Artifact Registry..."
gcloud auth configure-docker
docker push $IMAGE_NAME:latest

# --- Deploy to Cloud Run with ALL secrets and settings ---
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 4 \
    --min-instances 0 \
    --concurrency 80 \
    --timeout 300 \
    --set-env-vars="ENVIRONMENT=production"
# --- Get the Service URL ---
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "✅ DEPLOYMENT SUCCEEDED (ALMOST DONE!)"
echo "🌐 Your Service URL is: $SERVICE_URL"
echo ""
echo "📋 FINAL STEP REQUIRED: Update your Webhook!"
echo "Run the command below in your terminal to make the bot work:"
echo ""
echo "gcloud run services update $SERVICE_NAME --region $REGION --update-env-vars=\"TELEGRAM_WEBHOOK_URL=${SERVICE_URL}/webhook\""