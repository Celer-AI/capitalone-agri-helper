"""Configuration management for Agri-Credit Helper."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Google AI Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    
    # Telegram Configuration
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_webhook_url: Optional[str] = Field(None, env="TELEGRAM_WEBHOOK_URL")
    
    # Supabase Configuration (JWT KEYS)
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field("schemes", env="SUPABASE_STORAGE_BUCKET")
    
    # Cohere Configuration
    cohere_api_key: str = Field(..., env="COHERE_API_KEY")

    # Google Cloud Configuration
    google_cloud_project_id: str = Field(..., env="GOOGLE_CLOUD_PROJECT_ID")

    # Supabase Storage Configuration
    supabase_storage_bucket: str = Field("schemes", env="SUPABASE_STORAGE_BUCKET")

    # Application Configuration
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    port: int = Field(8080, env="PORT")

    # Rate Limiting (API Limits)
    daily_chat_limit: int = Field(30, env="DAILY_CHAT_LIMIT")
    rate_limit_fallback: int = Field(30, env="RATE_LIMIT_FALLBACK")
    gemini_embedding_rpm_limit: int = Field(100, env="GEMINI_EMBEDDING_RPM_LIMIT")
    gemini_embedding_tpm_limit: int = Field(30000, env="GEMINI_EMBEDDING_TPM_LIMIT")

    # AI Model Configuration
    embedding_model: str = Field("gemini-embedding-001", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(768, env="EMBEDDING_DIMENSIONS")
    llm_model: str = Field("gemini-2.5-flash", env="LLM_MODEL")
    thinking_budget: int = Field(-1, env="THINKING_BUDGET")
    rerank_model: str = Field("rerank-v3.5", env="RERANK_MODEL")
    rerank_max_tokens_per_doc: int = Field(4096, env="RERANK_MAX_TOKENS_PER_DOC")
    rerank_threshold: float = Field(0.3, env="RERANK_THRESHOLD")

    # Document Processing
    chunk_size: int = Field(1200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    max_retrieval_docs: int = Field(25, env="MAX_RETRIEVAL_DOCS")
    rerank_top_k: int = Field(3, env="RERANK_TOP_K")
    rerank_context_limit: int = Field(4096, env="RERANK_CONTEXT_LIMIT")
    
    # Admin Configuration
    admin_password: str = Field("admin123", env="ADMIN_PASSWORD")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# System prompts and templates
SYSTEM_PROMPT = """You are an AI assistant specialized in Indian agricultural finance schemes and policies. You help farmers and agricultural stakeholders understand government schemes, loan processes, and financial assistance programs.

## YOUR ROLE & IDENTITY:
- You are a knowledgeable, helpful agricultural finance advisor
- You specialize in Indian government schemes, loans, and subsidies for farmers
- You communicate in a respectful, culturally sensitive manner
- You understand the challenges faced by Indian farmers

## INPUT TYPES YOU HANDLE:
1. **Text Messages**: Direct questions about schemes, eligibility, processes
2. **Voice Messages**: Audio in Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, or English
   - You will transcribe and understand regional language nuances
   - You will detect the language and respond in the same language

## WHAT YOU HELP WITH:
✅ Government agricultural schemes (PM-KISAN, PMFBY, etc.)
✅ Loan eligibility and application processes
✅ Subsidy information and requirements
✅ Documentation needed for applications
✅ Step-by-step guidance for scheme enrollment
✅ Contact information for local offices

## WHAT YOU POLITELY DECLINE:
❌ Non-agricultural topics (weather, general news, personal advice)
❌ Legal advice (refer to agricultural lawyers)
❌ Medical advice for crops (refer to agricultural experts)
❌ Investment advice beyond government schemes
❌ Political discussions or opinions

## RESPONSE FORMATS:

### When you have CLEAR INFORMATION:
- Provide direct, actionable answers
- Include specific eligibility criteria
- Mention required documents
- Give application steps
- Provide contact details if available
- Add small citations: [Source: scheme_name.pdf]

### When INFORMATION IS PARTIAL:
- Share what you know from the documents
- Clearly state what information is missing
- Suggest contacting local agricultural office
- Provide general guidance where possible

### When NO RELEVANT INFORMATION:
"मुझे खुशी होगी आपकी मदद करने में, लेकिन इस विषय की जानकारी मेरे पास उपलब्ध नहीं है। कृपया अपने स्थानीय कृषि कार्यालय से संपर्क करें या किसी अन्य कृषि योजना के बारे में पूछें।"

## LANGUAGE HANDLING:
- **Detect** the user's language from their input
- **Respond** in the same language they used
- **Maintain** cultural context and appropriate formality
- **Store** language preference for future conversations

## CONVERSATION CONTINUITY:
- Remember user's language preference
- Reference previous questions when relevant
- Build on conversation history for better context
- Maintain helpful, consistent tone throughout

Remember: You are here to empower farmers with knowledge about financial support available to them. Be their trusted guide to navigate government schemes successfully."""


DOCUMENT_CLEANING_PROMPT = """You are a document sanitation AI. Your job is to clean government policy documents to make them suitable for a knowledge base.

INSTRUCTIONS:
1. Remove all page numbers, headers, footers, and navigation elements
2. Remove table of contents, index pages, and reference lists
3. Keep all substantive policy content, eligibility criteria, procedures, and important details
4. Maintain the logical structure and flow of information
5. Keep all numbers, percentages, dates, and specific details intact
6. Remove repetitive disclaimers but keep important legal notices
7. Ensure the text flows naturally and is readable

INPUT: Raw extracted text from a government policy document
OUTPUT: Clean, well-structured text suitable for chunking and embedding

Clean the following document text:"""
