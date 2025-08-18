#!/bin/bash

# Deploy and Test Script for Agri-Credit Helper
# This script deploys to Google Cloud Run and tests the deployment

set -e  # Exit on any error

echo "🚀 Starting deployment to Google Cloud Run..."

# Add Homebrew to PATH
export PATH="/opt/homebrew/bin:$PATH"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi

# Set project
PROJECT_ID="capitalone-468806"
REGION="asia-south1"
SERVICE_NAME="agri-credit-helper"

echo "📋 Using project: $PROJECT_ID"
echo "📍 Region: $REGION"
echo "🏷️  Service: $SERVICE_NAME"

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID

# Submit build
echo "🏗️  Starting Cloud Build..."
gcloud builds submit --config cloudbuild.yaml --project=$PROJECT_ID

# Wait for deployment to be ready
echo "⏳ Waiting for service to be ready..."
sleep 30

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format="value(status.url)")

if [ -z "$SERVICE_URL" ]; then
    echo "❌ Failed to get service URL"
    exit 1
fi

echo "🌐 Service URL: $SERVICE_URL"

# Test health endpoint
echo "🏥 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$SERVICE_URL/health")
HEALTH_CODE="${HEALTH_RESPONSE: -3}"

if [ "$HEALTH_CODE" = "200" ]; then
    echo "✅ Health check passed!"
    cat /tmp/health_response.json | python3 -m json.tool
else
    echo "❌ Health check failed with code: $HEALTH_CODE"
    cat /tmp/health_response.json
    exit 1
fi

# Test root endpoint
echo "🏠 Testing root endpoint..."
ROOT_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/root_response.json "$SERVICE_URL/")
ROOT_CODE="${ROOT_RESPONSE: -3}"

if [ "$ROOT_CODE" = "200" ]; then
    echo "✅ Root endpoint working!"
    cat /tmp/root_response.json | python3 -m json.tool
else
    echo "❌ Root endpoint failed with code: $ROOT_CODE"
    cat /tmp/root_response.json
fi

# Test query endpoint
echo "🔍 Testing query endpoint..."
QUERY_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/query_response.json -X POST "$SERVICE_URL/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is KCC?", "user_id": 12345}')
QUERY_CODE="${QUERY_RESPONSE: -3}"

if [ "$QUERY_CODE" = "200" ]; then
    echo "✅ Query endpoint working!"
    cat /tmp/query_response.json | python3 -m json.tool
else
    echo "⚠️  Query endpoint returned code: $QUERY_CODE"
    cat /tmp/query_response.json
fi

# Set up webhook URL (if deployment successful)
if [ "$HEALTH_CODE" = "200" ]; then
    WEBHOOK_URL="$SERVICE_URL/webhook"
    echo "🔗 Setting up Telegram webhook..."
    
    # Update the service with the webhook URL
    gcloud run services update $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --set-env-vars="TELEGRAM_WEBHOOK_URL=$WEBHOOK_URL"
    
    echo "✅ Webhook URL set to: $WEBHOOK_URL"
fi

echo ""
echo "🎉 Deployment Summary:"
echo "   Service URL: $SERVICE_URL"
echo "   Health: $HEALTH_CODE"
echo "   Root: $ROOT_CODE"
echo "   Query: $QUERY_CODE"
echo "   Admin Dashboard: $SERVICE_URL/admin"
echo ""

if [ "$HEALTH_CODE" = "200" ]; then
    echo "✅ Deployment successful! Your Agri-Credit Helper is live!"
else
    echo "❌ Deployment had issues. Check the logs:"
    echo "   gcloud logs read --project=$PROJECT_ID --service=$SERVICE_NAME"
fi

# Clean up temp files
rm -f /tmp/health_response.json /tmp/root_response.json /tmp/query_response.json
