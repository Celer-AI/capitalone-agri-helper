# 🌾 Agri-Credit Helper

An AI-powered conversational agent designed to help Indian farmers navigate government agricultural finance schemes and policies through Telegram.

## 🎯 Overview

Agri-Credit Helper bridges the information gap between farmers and complex government policy documents. Farmers can ask questions in their native language (Hindi, Tamil, English, etc.) via text or voice messages on Telegram and receive accurate, actionable information about agricultural finance schemes like KCC (Kisan Credit Card), PM-KISAN, and other government programs.

## ✨ Features

### For Farmers
- 🗣️ **Multilingual Support**: Ask questions in Hindi, Tamil, English, or mixed languages
- 🎤 **Voice Messages**: Send voice notes for hands-free interaction
- 📱 **Telegram Integration**: Use familiar Telegram interface
- 🎯 **Accurate Information**: Responses based on official government documents
- ⚡ **Real-time Responses**: Get answers within seconds
- 📊 **Rate Limiting**: 30 queries per day to ensure fair usage

### For Administrators
- 📄 **Document Upload**: Easy PDF upload through web interface
- 📝 **Text Input**: Direct text input bypassing PDF extraction
- 📊 **Analytics Dashboard**: Comprehensive usage analytics and insights
- 🔍 **System Monitoring**: Real-time system health and performance metrics
- 📈 **User Analytics**: Track user engagement and query patterns

## 🏗️ Architecture

### Core Components
1. **Telegram Bot**: Handles user interactions via webhook
2. **RAG Pipeline**: Retrieve-Augment-Generate pipeline for accurate responses
3. **Document Processor**: PDF processing and text chunking
4. **Admin Dashboard**: Streamlit-based management interface
5. **Vector Database**: Supabase with pg_vector for semantic search

### AI Models
- **LLM**: `gemini-2.5-flash` with dynamic thinking
- **Embeddings**: `gemini-embedding-001` (768 dimensions)
- **Reranking**: Cohere Rerank API for relevance optimization

### Infrastructure
- **Deployment**: Google Cloud Run (Mumbai region)
- **Database**: Supabase (PostgreSQL + Vector Store)
- **Storage**: Supabase Storage for document files
- **CI/CD**: GitHub Actions with automated deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Cloud Project
- Supabase Account
- Telegram Bot Token
- Cohere API Key

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agri-credit-helper
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Access the application**
   - Main API: http://localhost:8080
   - Admin Dashboard: http://localhost:8501
   - Health Check: http://localhost:8080/health

### Production Deployment

1. **Deploy to Google Cloud Run**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

2. **Set environment variables in Cloud Run**
   ```bash
   gcloud run services update agri-credit-helper \
     --region asia-south1 \
     --set-env-vars GEMINI_API_KEY=your_key,SUPABASE_URL=your_url
   ```

3. **Configure Telegram webhook**
   ```bash
   curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://your-service-url.run.app/webhook"}'
   ```

## 📊 Database Schema

### Core Tables
- **documents**: Stores document chunks with vector embeddings
- **users**: User information and rate limiting data
- **chat_history**: Conversation history for context
- **analytics**: System events and usage metrics

### Required SQL Function
Execute in Supabase SQL editor:
```sql
-- See sql/match_documents.sql for the complete function
```

## 🔧 Configuration

### Environment Variables
```bash
# Core API Keys
GEMINI_API_KEY=your_gemini_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
COHERE_API_KEY=your_cohere_api_key

# Application Settings
DAILY_CHAT_LIMIT=30
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
EMBEDDING_DIMENSIONS=768
```

### Model Configuration
- **Embedding Model**: `gemini-embedding-001` with 768 dimensions
- **LLM Model**: `gemini-2.5-flash` with dynamic thinking
- **Thinking Budget**: -1 (dynamic)
- **Rerank Model**: `rerank-english-v3.0`

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_rag_pipeline.py -v
pytest tests/test_document_processor.py -v
```

## 📈 Monitoring & Analytics

### Health Checks
- **Endpoint**: `/health`
- **Monitoring**: Built-in health checks for all services
- **Alerts**: Structured logging with error tracking

### Analytics Dashboard
Access comprehensive analytics at `/admin`:
- User engagement metrics
- Query success rates
- Language distribution
- System performance
- Document processing stats

## 🔒 Security & Rate Limiting

### Rate Limiting
- 30 queries per user per day
- Configurable limits via environment variables
- Automatic daily reset

### Security Features
- Admin dashboard password protection
- Environment-based configuration
- Structured logging for audit trails
- Error handling without data exposure

## 🌍 Multilingual Support

### Supported Languages
- Hindi (हिंदी)
- Tamil (தமிழ்)
- English
- Bengali (বাংলা)
- Marathi (मराठी)
- And other Indian languages

### Language Detection
- Automatic language detection from user input
- Response generation in the same language
- Support for mixed languages (Hinglish)

## 📚 API Documentation

### Main Endpoints
- `GET /` - Service information
- `GET /health` - Health check
- `POST /webhook` - Telegram webhook
- `POST /query` - Direct query endpoint (testing)
- `GET /stats` - Application statistics

### Admin Endpoints
- `POST /admin/upload` - Document upload
- Streamlit dashboard at `/admin`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### For Technical Issues
- Check the health endpoint: `/health`
- Review logs in Google Cloud Console
- Monitor Supabase dashboard

### For Farmers
- Telegram bot commands: `/help`, `/status`
- Kisan Call Center: 1800-180-1551
- Local agriculture department offices

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Core RAG pipeline
- ✅ Telegram integration
- ✅ Admin dashboard
- ✅ Cloud deployment

### Phase 2 (Future)
- [ ] Advanced analytics
- [ ] Multi-tenant support
- [ ] API rate limiting improvements
- [ ] Enhanced document processing
- [ ] Mobile app integration

---

**Built with ❤️ for Indian farmers**
