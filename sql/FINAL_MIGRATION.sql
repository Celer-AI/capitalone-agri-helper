-- ============================================================================
--      CAPITAL ONE AGRI-CREDIT HELPER - CONSOLIDATED MIGRATION SCRIPT
-- This single, idempotent script handles all database setup and migrations.
-- It can be safely run multiple times without causing errors.
-- ============================================================================

-- Section 1: Extensions
-- Ensure necessary PostgreSQL extensions are enabled.
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Section 2: Table Creation
-- Defines the entire schema, creating tables only if they do not exist.
-- All columns are included, removing the need for subsequent ALTER TABLE statements.
-- ============================================================================

-- Documents Table: Stores text chunks and their vector embeddings for similarity search.
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

-- Users Table: Stores information about Telegram users interacting with the bot.
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

-- Chat History Table: Logs all interactions between users and the assistant.
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    telegram_user_id BIGINT REFERENCES users(telegram_user_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    message_type TEXT DEFAULT 'text' CHECK (message_type IN ('text', 'voice')),
    language_detected TEXT,
    response_time_ms INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analytics Table: Stores events for tracking and analysis purposes.
CREATE TABLE IF NOT EXISTS analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL,
    user_id BIGINT,
    session_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document Processing Logs Table: Tracks the status of document ingestion.
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

-- System Metrics Table: Stores various system-level performance metrics.
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_type TEXT DEFAULT 'counter' CHECK (metric_type IN ('counter', 'gauge', 'histogram')),
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Section 3: Index Creation
-- Creates indexes for performance, only if they do not already exist.
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
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
-- Section 4: Functions
-- Creates or replaces all necessary SQL functions.
-- ============================================================================

-- Function for vector similarity search.
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
LANGUAGE plpgsql STABLE PARALLEL SAFE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.source_document,
        d.chunk_index,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents AS d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY (d.embedding <=> query_embedding) ASC
    LIMIT match_count;
END;
$$;

-- Function to reset daily chat counts for all users.
CREATE OR REPLACE FUNCTION reset_daily_chat_counts()
RETURNS INTEGER
LANGUAGE sql
AS $$
    UPDATE users
    SET daily_chat_count = 0,
        last_reset_date = CURRENT_DATE,
        updated_at = NOW()
    WHERE last_reset_date < CURRENT_DATE;
    SELECT COUNT(*)::INTEGER FROM users WHERE last_reset_date = CURRENT_DATE;
$$;

-- Function to retrieve statistics for a specific user.
CREATE OR REPLACE FUNCTION get_user_stats(p_user_id BIGINT)
RETURNS TABLE(
    total_messages BIGINT,
    messages_today BIGINT,
    first_interaction TIMESTAMPTZ,
    last_interaction TIMESTAMPTZ,
    avg_response_time_ms DOUBLE PRECISION
)
LANGUAGE sql STABLE
AS $$
    SELECT
        COUNT(id),
        COUNT(id) FILTER (WHERE created_at::date = CURRENT_DATE),
        MIN(created_at),
        MAX(created_at),
        AVG(response_time_ms)
    FROM chat_history
    WHERE telegram_user_id = p_user_id;
$$;

-- Function to clean up chat history older than a specified number of days.
CREATE OR REPLACE FUNCTION cleanup_old_chat_history(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER
LANGUAGE sql
AS $$
    WITH deleted AS (
        DELETE FROM chat_history
        WHERE created_at < NOW() - (days_to_keep * INTERVAL '1 day')
        RETURNING id
    )
    SELECT COUNT(*)::INTEGER FROM deleted;
$$;

-- Trigger function to automatically update the 'updated_at' timestamp.
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Section 5: Triggers
-- Attaches the timestamp update function to relevant tables.
-- ============================================================================
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Section 6: Permissions
-- Grants necessary privileges to API roles for seamless operation.
-- ============================================================================
GRANT USAGE ON SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres, anon, authenticated, service_role;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres, anon, authenticated, service_role;

-- ============================================================================
-- Final Verification
-- ============================================================================
SELECT 'CAPITAL ONE AGRI-CREDIT HELPER MIGRATION COMPLETED SUCCESSFULLY!' AS status;