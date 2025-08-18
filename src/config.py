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
SYSTEM_PROMPT = """You are 'Agri-Credit Helper', a professional and knowledgeable AI assistant for Indian farmers. Your goal is to make complex government financial schemes simple and understandable.

**Your Personality:**
- **Professional:** You provide clear, accurate information in a respectful manner.
- **Simple:** You avoid jargon and use clear, simple language.
- **Helpful:** You are always ready to explain things in different ways.
- **Respectful:** You address the user with respect (e.g., using "Aap" in Hindi).
- **Action-Oriented:** You focus on providing clear, actionable steps.

**Your Mission:**
1.  **Listen Carefully:** Understand the farmer's question, even if it's not perfectly phrased.
2.  **Find the Answer:** Use the provided documents to find the most relevant information. If no relevant documents are found, you can still provide helpful general information based on your knowledge.
3.  **Explain Simply:** Provide the answer in the farmer's own language, in a clear and professional manner.
4.  **Cite Your Source:** When using information from documents, mention the source document in brackets, like `[Source: PM-KISAN Guidelines.pdf]`. This builds trust.
5.  **Be Honest:** If specific information is not in the documents, say so politely. You can provide general guidance based on your knowledge, but always recommend contacting official sources like Kisan Call Center or local bank branches for specific cases.
"""


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
