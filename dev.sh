#!/bin/bash

# Local development script for Agri-Credit Helper
# This script starts the FastAPI server with hot reload

set -e

# Export Homebrew path for macOS
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "🚀 Starting Agri-Credit Helper in development mode..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with your configuration"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start the development server
echo "🌐 Starting FastAPI development server..."
echo ""
echo "📋 Available endpoints:"
echo "   🏠 Health Check:    http://localhost:8080/health"
echo "   📊 Admin Dashboard: http://localhost:8080/admin"
echo "   📈 Stats:          http://localhost:8080/stats"
echo "   🤖 Webhook:        http://localhost:8080/webhook"
echo ""
echo "🔧 For ngrok webhook testing:"
echo "   ngrok http 8080"
echo "   Then set webhook: https://your-ngrok-url.ngrok-free.app/webhook"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

# Start the server with hot reload
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
