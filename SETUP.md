# 🚀 Setup Guide for Agri-Credit Helper

This guide will walk you through setting up the Agri-Credit Helper from scratch.

## 📋 Prerequisites

### Required Accounts & Services
1. **Google Cloud Platform Account**
   - Enable billing
   - Create a new project or use existing one

2. **Supabase Account**
   - Create a new project
   - Note down the URL and API keys

3. **Telegram Bot**
   - Create bot via @BotFather
   - Get bot token

4. **Cohere Account**
   - Sign up for Cohere API
   - Get API key

### Required Tools
- Python 3.11+
- Docker & Docker Compose
- Google Cloud CLI
- Git

## 🔧 Step-by-Step Setup

### 1. Google Cloud Setup

```bash
# Install Google Cloud CLI (if not installed)
# Follow: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Supabase Setup

1. **Create Supabase Project**
   - Go to https://supabase.com
   - Create new project
   - Wait for setup to complete

2. **Enable pg_vector Extension**
   ```sql
   -- In Supabase SQL Editor
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Create the similarity search function**
   ```sql
   -- Copy and execute the content from sql/match_documents.sql
   ```

4. **Get API Keys**
   - Go to Settings > API
   - Copy `anon` key and `service_role` key
   - Note the project URL

### 3. Telegram Bot Setup

1. **Create Bot**
   - Message @BotFather on Telegram
   - Send `/newbot`
   - Follow instructions to create bot
   - Save the bot token

2. **Configure Bot**
   ```bash
   # Set bot commands (optional)
   curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setMyCommands" \
   -H "Content-Type: application/json" \
   -d '{
     "commands": [
       {"command": "start", "description": "Start the bot"},
       {"command": "help", "description": "Get help"},
       {"command": "status", "description": "Check your status"}
     ]
   }'
   ```

### 4. Cohere API Setup

1. **Get API Key**
   - Go to https://cohere.ai
   - Sign up/login
   - Go to API Keys section
   - Create new API key

### 5. Local Development Setup

1. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd agri-credit-helper
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```

3. **Edit .env file**
   ```bash
   # Google AI Configuration
   GEMINI_API_KEY=your_gemini_api_key_here

   # Telegram Configuration
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   TELEGRAM_WEBHOOK_URL=https://your-cloud-run-url.run.app/webhook

   # Supabase Configuration
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your_supabase_anon_key_here
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here

   # Cohere Configuration
   COHERE_API_KEY=your_cohere_api_key_here

   # Application Configuration
   ENVIRONMENT=development
   DAILY_CHAT_LIMIT=30
   ADMIN_PASSWORD=your_secure_password_here
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Locally**
   ```bash
   # Option 1: Direct Python
   python -m uvicorn src.main:app --reload --port 8080

   # Option 2: Docker Compose
   docker-compose up -d
   ```

### 6. Production Deployment

1. **Set Environment Variables for Deployment**
   ```bash
   export PROJECT_ID=your-gcp-project-id
   export REGION=asia-south1
   ```

2. **Deploy to Cloud Run**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Set Production Environment Variables**
   ```bash
   gcloud run services update agri-credit-helper \
     --region asia-south1 \
     --set-env-vars \
     GEMINI_API_KEY=your_key,\
     TELEGRAM_BOT_TOKEN=your_token,\
     SUPABASE_URL=your_url,\
     SUPABASE_KEY=your_key,\
     SUPABASE_SERVICE_ROLE_KEY=your_service_key,\
     COHERE_API_KEY=your_cohere_key,\
     DAILY_CHAT_LIMIT=30,\
     ADMIN_PASSWORD=your_secure_password
   ```

4. **Get Service URL**
   ```bash
   gcloud run services describe agri-credit-helper \
     --region asia-south1 \
     --format 'value(status.url)'
   ```

### 7. Configure Telegram Webhook

```bash
# Replace <YOUR_BOT_TOKEN> and <YOUR_SERVICE_URL>
curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url": "<YOUR_SERVICE_URL>/webhook"}'

# Verify webhook
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getWebhookInfo"
```

### 8. Initial Data Setup

1. **Access Admin Dashboard**
   - Go to `<YOUR_SERVICE_URL>/admin` (for production)
   - Or `http://localhost:8501` (for local)

2. **Upload Initial Documents**
   - Use the Document Upload tab
   - Upload 2-3 key government policy PDFs
   - Verify successful processing

3. **Test the System**
   - Message your Telegram bot
   - Send `/start` to begin
   - Ask a test question about agricultural schemes

## 🔍 Verification Steps

### 1. Health Checks
```bash
# Check main service
curl <YOUR_SERVICE_URL>/health

# Check specific endpoints
curl <YOUR_SERVICE_URL>/stats
```

### 2. Database Verification
```sql
-- In Supabase SQL Editor
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM analytics;
```

### 3. Telegram Bot Testing
- Send `/start` to your bot
- Send a voice message
- Send a text question about KCC or PM-KISAN
- Verify responses are in correct language

### 4. Admin Dashboard Testing
- Access admin dashboard
- Upload a test PDF
- Check analytics data
- Verify system status

## 🚨 Troubleshooting

### Common Issues

1. **Telegram Webhook Not Working**
   ```bash
   # Check webhook status
   curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getWebhookInfo"
   
   # Reset webhook
   curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/deleteWebhook"
   # Then set it again
   ```

2. **Database Connection Issues**
   - Verify Supabase URL and keys
   - Check if pg_vector extension is enabled
   - Ensure similarity search function is created

3. **AI API Issues**
   - Verify all API keys are correct
   - Check API quotas and limits
   - Monitor Cloud Run logs

4. **Document Processing Failures**
   - Check PDF file format
   - Verify text extraction is working
   - Monitor embedding generation

### Monitoring Commands

```bash
# View Cloud Run logs
gcloud logs read --service=agri-credit-helper --region=asia-south1

# Check service status
gcloud run services describe agri-credit-helper --region=asia-south1

# Monitor resource usage
gcloud run services describe agri-credit-helper --region=asia-south1 --format="value(status.traffic)"
```

## 📊 Performance Optimization

### Recommended Settings
- **Memory**: 2Gi
- **CPU**: 2
- **Max Instances**: 10
- **Min Instances**: 1
- **Concurrency**: 80

### Monitoring Metrics
- Response time < 5 seconds
- Success rate > 95%
- Memory usage < 80%
- CPU usage < 70%

## 🔒 Security Checklist

- [ ] All API keys stored as environment variables
- [ ] Admin dashboard password set
- [ ] Supabase RLS policies configured (if needed)
- [ ] Cloud Run service properly configured
- [ ] Webhook URL uses HTTPS
- [ ] Rate limiting enabled

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Cloud Run logs
3. Verify all environment variables
4. Test individual components
5. Check API quotas and limits

---

**Setup complete! Your Agri-Credit Helper is ready to help farmers! 🌾**
