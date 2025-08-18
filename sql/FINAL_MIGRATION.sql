-- ============================================================================
-- FINAL PRODUCTION MIGRATION - Capital One Agri-Credit Helper
-- This file handles EVERYTHING and works regardless of current state
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- DROP EXISTING FUNCTIONS TO AVOID CONFLICTS
-- ============================================================================
DROP FUNCTION IF EXISTS match_documents(vector, double precision, integer);
DROP FUNCTION IF EXISTS match_documents(vector, float, int);
DROP FUNCTION IF EXISTS reset_daily_chat_counts();
DROP FUNCTION IF EXISTS get_user_stats(bigint);
DROP FUNCTION IF EXISTS cleanup_old_chat_history(integer);

-- ============================================================================
-- CREATE TABLES WITH IF NOT EXISTS
-- ============================================================================

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    source_document TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    telegram_user_id BIGINT PRIMARY KEY,
    username TEXT,
    first_name TEXT,
    last_name TEXT,
    language_code TEXT DEFAULT 'hi',
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    daily_chat_count INTEGER DEFAULT 0,
    last_reset_date DATE DEFAULT CURRENT_DATE,
    total_queries INTEGER DEFAULT 0,
    is_blocked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat history table
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    telegram_user_id BIGINT REFERENCES users(telegram_user_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    message_type TEXT DEFAULT 'text' CHECK (message_type IN ('text', 'voice')),
    language_detected TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analytics table
CREATE TABLE IF NOT EXISTS analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL,
    user_id BIGINT,
    session_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document processing logs table
CREATE TABLE IF NOT EXISTS document_processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    source_type TEXT DEFAULT 'pdf' CHECK (source_type IN ('pdf', 'text')),
    status TEXT DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed')),
    chunks_created INTEGER DEFAULT 0,
    chunks_stored INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_type TEXT DEFAULT 'counter' CHECK (metric_type IN ('counter', 'gauge', 'histogram')),
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- ADD MISSING COLUMNS TO EXISTING TABLES
-- ============================================================================

-- Add missing columns to documents table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='chunk_index') THEN
        ALTER TABLE documents ADD COLUMN chunk_index INTEGER DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='metadata') THEN
        ALTER TABLE documents ADD COLUMN metadata JSONB DEFAULT '{}';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='updated_at') THEN
        ALTER TABLE documents ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;

-- Add missing columns to users table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='username') THEN
        ALTER TABLE users ADD COLUMN username TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='first_name') THEN
        ALTER TABLE users ADD COLUMN first_name TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='last_name') THEN
        ALTER TABLE users ADD COLUMN last_name TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='language_code') THEN
        ALTER TABLE users ADD COLUMN language_code TEXT DEFAULT 'hi';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='total_queries') THEN
        ALTER TABLE users ADD COLUMN total_queries INTEGER DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='is_blocked') THEN
        ALTER TABLE users ADD COLUMN is_blocked BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='created_at') THEN
        ALTER TABLE users ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='updated_at') THEN
        ALTER TABLE users ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;

-- Add missing columns to chat_history table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chat_history' AND column_name='message_type') THEN
        ALTER TABLE chat_history ADD COLUMN message_type TEXT DEFAULT 'text' CHECK (message_type IN ('text', 'voice'));
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chat_history' AND column_name='language_detected') THEN
        ALTER TABLE chat_history ADD COLUMN language_detected TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chat_history' AND column_name='response_time_ms') THEN
        ALTER TABLE chat_history ADD COLUMN response_time_ms INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chat_history' AND column_name='metadata') THEN
        ALTER TABLE chat_history ADD COLUMN metadata JSONB DEFAULT '{}';
    END IF;
END $$;

-- Add missing columns to analytics table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='analytics' AND column_name='session_id') THEN
        ALTER TABLE analytics ADD COLUMN session_id TEXT;
    END IF;
END $$;

-- ============================================================================
-- CREATE INDEXES (IF NOT EXISTS)
-- ============================================================================

-- Vector similarity search index
CREATE INDEX IF NOT EXISTS idx_documents_embedding 
ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Other indexes
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_document);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_users_daily_count ON users(daily_chat_count);
CREATE INDEX IF NOT EXISTS idx_users_last_reset ON users(last_reset_date);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(telegram_user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_recent ON chat_history(telegram_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_document_logs_status ON document_processing_logs(status);
CREATE INDEX IF NOT EXISTS idx_document_logs_created_at ON document_processing_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_created_at ON system_metrics(created_at DESC);

-- ============================================================================
-- CREATE FUNCTIONS
-- ============================================================================

-- Vector similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE(
    id UUID,
    content TEXT,
    source_document TEXT,
    chunk_index INTEGER,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        d.id,
        d.content,
        d.source_document,
        d.chunk_index,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Daily reset function
CREATE OR REPLACE FUNCTION reset_daily_chat_counts()
RETURNS INTEGER
LANGUAGE SQL
AS $$
    UPDATE users 
    SET daily_chat_count = 0, 
        last_reset_date = CURRENT_DATE,
        updated_at = NOW()
    WHERE last_reset_date < CURRENT_DATE;
    
    SELECT COUNT(*)::INTEGER FROM users WHERE last_reset_date = CURRENT_DATE;
$$;

-- User stats function
CREATE OR REPLACE FUNCTION get_user_stats(user_id BIGINT)
RETURNS TABLE(
    total_messages INTEGER,
    messages_today INTEGER,
    first_interaction TIMESTAMPTZ,
    last_interaction TIMESTAMPTZ,
    avg_response_time FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        COUNT(*)::INTEGER as total_messages,
        COUNT(CASE WHEN DATE(created_at) = CURRENT_DATE THEN 1 END)::INTEGER as messages_today,
        MIN(created_at) as first_interaction,
        MAX(created_at) as last_interaction,
        AVG(response_time_ms)::FLOAT as avg_response_time
    FROM chat_history
    WHERE telegram_user_id = user_id;
$$;

-- Cleanup function
CREATE OR REPLACE FUNCTION cleanup_old_chat_history(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER
LANGUAGE SQL
AS $$
    WITH deleted AS (
        DELETE FROM chat_history 
        WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep
        RETURNING id
    )
    SELECT COUNT(*)::INTEGER FROM deleted;
$$;

-- ============================================================================
-- CREATE TRIGGERS
-- ============================================================================

-- Update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FINAL VERIFICATION
-- ============================================================================
SELECT 'MIGRATION COMPLETED SUCCESSFULLY!' as status;
