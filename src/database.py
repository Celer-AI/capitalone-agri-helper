"""Database operations and schema management."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
import numpy as np
from supabase import create_client, Client
from src.config import settings
import structlog

logger = structlog.get_logger()


class DatabaseManager:
    """Manages all database operations for the Agri-Credit Helper."""

    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )
    
    async def initialize_schema(self):
        """Initialize database schema with required tables and extensions."""
        try:
            # Enable required extensions
            await self._execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")
            await self._execute_sql('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

            # Create documents table with all required columns
            documents_sql = f"""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                content TEXT NOT NULL,
                embedding VECTOR({settings.embedding_dimensions}) NOT NULL,
                source_document TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
            await self._execute_sql(documents_sql)

            # Create users table with all required columns
            users_sql = """
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
            """
            await self._execute_sql(users_sql)

            # Create chat_history table with all required columns
            chat_history_sql = """
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
            """
            await self._execute_sql(chat_history_sql)

            # Create analytics table with all required columns
            analytics_sql = """
            CREATE TABLE IF NOT EXISTS analytics (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                event_type TEXT NOT NULL,
                user_id BIGINT,
                session_id TEXT,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
            await self._execute_sql(analytics_sql)

            # Create document_processing_logs table
            doc_logs_sql = """
            CREATE TABLE IF NOT EXISTS document_processing_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                filename TEXT NOT NULL,
                source_type TEXT DEFAULT 'pdf' CHECK (source_type IN ('pdf', 'text')),
                status TEXT DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed')),
                chunks_created INTEGER DEFAULT 0,
                chunks_stored INTEGER DEFAULT 0,
                processing_time_ms INTEGER,
                error_message TEXT,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            );
            """
            await self._execute_sql(doc_logs_sql)

            # Create system_metrics table
            metrics_sql = """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                metric_name TEXT NOT NULL,
                metric_value NUMERIC NOT NULL,
                metric_type TEXT DEFAULT 'counter' CHECK (metric_type IN ('counter', 'gauge', 'histogram')),
                tags JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
            await self._execute_sql(metrics_sql)
            
            # Create indexes for better performance
            indexes = [
                # Vector similarity search index (IVFFlat for better performance)
                f"CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",

                # Regular indexes for common queries
                "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_document);",
                "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);",

                "CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen DESC);",
                "CREATE INDEX IF NOT EXISTS idx_users_daily_count ON users(daily_chat_count);",
                "CREATE INDEX IF NOT EXISTS idx_users_last_reset ON users(last_reset_date);",

                "CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(telegram_user_id);",
                "CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_chat_history_user_recent ON chat_history(telegram_user_id, created_at DESC);",

                "CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);",
                "CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id);",

                "CREATE INDEX IF NOT EXISTS idx_document_logs_status ON document_processing_logs(status);",
                "CREATE INDEX IF NOT EXISTS idx_document_logs_created_at ON document_processing_logs(created_at DESC);",

                "CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);",
                "CREATE INDEX IF NOT EXISTS idx_system_metrics_created_at ON system_metrics(created_at DESC);"
            ]

            for index_sql in indexes:
                await self._execute_sql(index_sql)

            # Create utility functions
            await self._create_utility_functions()

            # Create triggers for automatic updates
            await self._create_triggers()

            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database schema", error=str(e))
            raise

    async def _create_utility_functions(self):
        """Create utility functions for database operations."""
        # Vector similarity search function
        match_documents_sql = f"""
        CREATE OR REPLACE FUNCTION match_documents(
            query_embedding VECTOR({settings.embedding_dimensions}),
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
        """
        await self._execute_sql(match_documents_sql)

        # Daily chat count reset function
        reset_counts_sql = """
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
        """
        await self._execute_sql(reset_counts_sql)

        # User statistics function
        user_stats_sql = """
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
        """
        await self._execute_sql(user_stats_sql)

        # Cleanup old chat history function
        cleanup_sql = """
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
        """
        await self._execute_sql(cleanup_sql)

    async def _create_triggers(self):
        """Create triggers for automatic timestamp updates."""
        # Update trigger function
        trigger_function_sql = """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """
        await self._execute_sql(trigger_function_sql)

        # Create triggers for tables with updated_at columns
        triggers = [
            "CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
            "CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();"
        ]

        for trigger_sql in triggers:
            await self._execute_sql(trigger_sql.replace("CREATE TRIGGER", "CREATE TRIGGER IF NOT EXISTS"))

    async def _execute_sql(self, sql: str):
        """Execute raw SQL command."""
        try:
            # Try using RPC first
            result = self.client.rpc('exec_sql', {'sql': sql}).execute()
            return result
        except Exception as e:
            logger.warning("RPC SQL execution failed, trying direct query", error=str(e))
            try:
                # Fallback: try a simple query to test connection
                if sql.strip().upper().startswith('SELECT'):
                    # For SELECT queries, try using the query method
                    result = self.client.from_('information_schema.tables').select('table_name').limit(1).execute()
                    return result
                else:
                    # For other queries, we need to use a different approach
                    logger.error("Cannot execute non-SELECT SQL without RPC support")
                    raise e
            except Exception as e2:
                logger.error("All SQL execution methods failed", original_error=str(e), fallback_error=str(e2))
                raise e
    
    async def upload_document_to_storage(self, file_content: bytes, filename: str) -> Optional[str]:
        """Upload document to Supabase Storage and return the public URL."""
        try:
            # Generate unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4()}_{filename}"

            # Upload to Supabase Storage
            result = self.client.storage.from_(settings.supabase_storage_bucket).upload(
                unique_filename,
                file_content,
                file_options={"content-type": "application/pdf"}
            )

            if result.error:
                logger.error("Failed to upload document to storage",
                           filename=filename,
                           error=result.error)
                return None

            # Get public URL
            public_url = self.client.storage.from_(settings.supabase_storage_bucket).get_public_url(unique_filename)

            logger.info("Document uploaded to storage successfully",
                       filename=filename,
                       storage_path=unique_filename)

            return public_url

        except Exception as e:
            logger.error("Failed to upload document to storage", filename=filename, error=str(e))
            return None

    async def store_document_chunks(self, chunks: List[Dict[str, Any]], storage_url: Optional[str] = None) -> bool:
        """Store document chunks with embeddings in the database."""
        try:
            # Prepare data for insertion
            documents_data = []
            for i, chunk in enumerate(chunks):
                documents_data.append({
                    'id': str(uuid.uuid4()),
                    'content': chunk['content'],
                    'embedding': chunk['embedding'],
                    'source_document': chunk['source_document'],
                    'chunk_index': i,
                    'metadata': {
                        'storage_url': storage_url,
                        'chunk_length': len(chunk['content']),
                        'processing_timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    'created_at': datetime.now(timezone.utc).isoformat()
                })

            # Insert in batches
            batch_size = 50  # Reduced batch size for better reliability
            total_inserted = 0

            for i in range(0, len(documents_data), batch_size):
                batch = documents_data[i:i + batch_size]
                result = self.client.table('documents').insert(batch).execute()

                if result.data:
                    total_inserted += len(result.data)
                    logger.debug("Inserted document batch",
                               batch_start=i,
                               batch_size=len(batch),
                               total_inserted=total_inserted)
                else:
                    logger.error("Failed to insert document batch",
                               batch_start=i,
                               error=getattr(result, 'error', 'Unknown error'))
                    return False

            logger.info("Successfully stored document chunks",
                       total_chunks=len(chunks),
                       total_inserted=total_inserted)
            return True

        except Exception as e:
            logger.error("Failed to store document chunks", error=str(e))
            return False
    
    async def similarity_search(self, query_embedding: List[float], limit: int = 25) -> List[Dict[str, Any]]:
        """Perform similarity search using vector embeddings."""
        try:
            # Convert embedding to the format expected by Supabase
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Perform similarity search
            result = self.client.rpc(
                'match_documents',
                {
                    'query_embedding': embedding_str,
                    'match_threshold': 0.1,
                    'match_count': limit
                }
            ).execute()
            
            if result.data:
                return result.data
            else:
                logger.warning("No similar documents found")
                return []
                
        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            return []
    
    async def get_user_chat_count(self, telegram_user_id: int) -> int:
        """Get current daily chat count for a user."""
        try:
            # First, reset count if it's a new day
            await self._reset_daily_count_if_needed(telegram_user_id)
            
            result = self.client.table('users').select('daily_chat_count').eq('telegram_user_id', telegram_user_id).execute()
            
            if result.data:
                return result.data[0]['daily_chat_count']
            else:
                # Create new user
                await self._create_user(telegram_user_id)
                return 0
                
        except Exception as e:
            logger.error("Failed to get user chat count", user_id=telegram_user_id, error=str(e))
            return 0
    
    async def increment_user_chat_count(self, telegram_user_id: int) -> bool:
        """Increment user's daily chat count."""
        try:
            current_count = await self.get_user_chat_count(telegram_user_id)
            new_count = current_count + 1
            
            result = self.client.table('users').update({
                'daily_chat_count': new_count,
                'last_seen': datetime.now(timezone.utc).isoformat()
            }).eq('telegram_user_id', telegram_user_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error("Failed to increment user chat count", user_id=telegram_user_id, error=str(e))
            return False
    
    async def store_chat_message(self, telegram_user_id: int, role: str, content: str,
                               message_type: str = 'text', language_detected: Optional[str] = None,
                               response_time_ms: Optional[int] = None,
                               transcription_data: Optional[Dict] = None):
        """Store chat message in history with enhanced metadata and transcription data."""
        try:
            # Prepare metadata for storage
            metadata = {}
            if transcription_data:
                metadata['transcription'] = transcription_data

            result = self.client.table('chat_history').insert({
                'telegram_user_id': telegram_user_id,
                'role': role,
                'content': content,
                'message_type': message_type,
                'language_detected': language_detected,
                'response_time_ms': response_time_ms,
                'metadata': metadata,
                'created_at': datetime.now(timezone.utc).isoformat()
            }).execute()

            # Return the chat ID for linking transcription
            if result.data:
                return result.data[0]['id']
            return None

        except Exception as e:
            logger.error("Failed to store chat message", user_id=telegram_user_id, error=str(e))
            return None
    
    async def get_chat_history(self, telegram_user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent chat history for a user with enhanced metadata."""
        try:
            result = self.client.table('chat_history').select(
                'id, role, content, message_type, language_detected, metadata, created_at'
            ).eq(
                'telegram_user_id', telegram_user_id
            ).order('created_at', desc=True).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error("Failed to get chat history", user_id=telegram_user_id, error=str(e))
            return []

    async def get_user_language_preference(self, telegram_user_id: int) -> Optional[str]:
        """Get user's preferred language from recent chat history."""
        try:
            result = self.client.table('chat_history').select(
                'language_detected'
            ).eq(
                'telegram_user_id', telegram_user_id
            ).eq(
                'role', 'user'
            ).not_.is_(
                'language_detected', 'null'
            ).order(
                'created_at', desc=True
            ).limit(1).execute()

            if result.data and result.data[0]['language_detected']:
                return result.data[0]['language_detected']

            # Fallback to user profile language
            user_result = self.client.table('users').select(
                'language_code'
            ).eq('telegram_user_id', telegram_user_id).execute()

            if user_result.data:
                return user_result.data[0]['language_code']

            return 'Hindi'  # Default fallback

        except Exception as e:
            logger.error("Failed to get user language preference", user_id=telegram_user_id, error=str(e))
            return 'Hindi'
    
    async def log_analytics_event(self, event_type: str, user_id: Optional[int] = None, metadata: Optional[Dict] = None):
        """Log analytics event."""
        try:
            result = self.client.table('analytics').insert({
                'event_type': event_type,
                'user_id': user_id,
                'metadata': metadata or {},
                'created_at': datetime.now(timezone.utc).isoformat()
            }).execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error("Failed to log analytics event", event_type=event_type, error=str(e))
            return False
    
    async def _reset_daily_count_if_needed(self, telegram_user_id: int):
        """Reset daily chat count if it's a new day."""
        try:
            result = self.client.table('users').select('last_reset_date').eq('telegram_user_id', telegram_user_id).execute()
            
            if result.data:
                last_reset = result.data[0]['last_reset_date']
                today = datetime.now(timezone.utc).date()
                
                if str(last_reset) != str(today):
                    # Reset count for new day
                    self.client.table('users').update({
                        'daily_chat_count': 0,
                        'last_reset_date': today.isoformat()
                    }).eq('telegram_user_id', telegram_user_id).execute()
                    
        except Exception as e:
            logger.error("Failed to reset daily count", user_id=telegram_user_id, error=str(e))
    
    async def _create_user(self, telegram_user_id: int, username: Optional[str] = None,
                         first_name: Optional[str] = None, last_name: Optional[str] = None,
                         language_code: str = 'hi'):
        """Create a new user record with enhanced profile information."""
        try:
            result = self.client.table('users').insert({
                'telegram_user_id': telegram_user_id,
                'username': username,
                'first_name': first_name,
                'last_name': last_name,
                'language_code': language_code,
                'daily_chat_count': 0,
                'total_queries': 0,
                'is_blocked': False,
                'last_seen': datetime.now(timezone.utc).isoformat(),
                'last_reset_date': datetime.now(timezone.utc).date().isoformat(),
                'created_at': datetime.now(timezone.utc).isoformat()
            }).execute()

            return bool(result.data)

        except Exception as e:
            logger.error("Failed to create user", user_id=telegram_user_id, error=str(e))
            return False

    async def update_user_profile(self, telegram_user_id: int, username: Optional[str] = None,
                                first_name: Optional[str] = None, last_name: Optional[str] = None,
                                language_code: Optional[str] = None):
        """Update user profile information."""
        try:
            update_data = {'last_seen': datetime.now(timezone.utc).isoformat()}

            if username is not None:
                update_data['username'] = username
            if first_name is not None:
                update_data['first_name'] = first_name
            if last_name is not None:
                update_data['last_name'] = last_name
            if language_code is not None:
                update_data['language_code'] = language_code

            result = self.client.table('users').update(update_data).eq(
                'telegram_user_id', telegram_user_id
            ).execute()

            return bool(result.data)

        except Exception as e:
            logger.error("Failed to update user profile", user_id=telegram_user_id, error=str(e))
            return False

    async def log_document_processing(self, filename: str, source_type: str = 'pdf',
                                    chunks_created: int = 0, chunks_stored: int = 0,
                                    processing_time_ms: Optional[int] = None,
                                    error_message: Optional[str] = None,
                                    metadata: Optional[Dict] = None) -> Optional[str]:
        """Log document processing operation."""
        try:
            status = 'completed' if error_message is None else 'failed'

            result = self.client.table('document_processing_logs').insert({
                'filename': filename,
                'source_type': source_type,
                'status': status,
                'chunks_created': chunks_created,
                'chunks_stored': chunks_stored,
                'processing_time_ms': processing_time_ms,
                'error_message': error_message,
                'metadata': metadata or {},
                'created_at': datetime.now(timezone.utc).isoformat(),
                'completed_at': datetime.now(timezone.utc).isoformat() if status == 'completed' else None
            }).execute()

            if result.data:
                return result.data[0]['id']
            return None

        except Exception as e:
            logger.error("Failed to log document processing", filename=filename, error=str(e))
            return None

    async def log_system_metric(self, metric_name: str, metric_value: float,
                              metric_type: str = 'counter', tags: Optional[Dict] = None):
        """Log system metric for monitoring."""
        try:
            result = self.client.table('system_metrics').insert({
                'metric_name': metric_name,
                'metric_value': metric_value,
                'metric_type': metric_type,
                'tags': tags or {},
                'created_at': datetime.now(timezone.utc).isoformat()
            }).execute()

            return bool(result.data)

        except Exception as e:
            logger.error("Failed to log system metric", metric_name=metric_name, error=str(e))
            return False


# Global database manager instance
db = DatabaseManager()


# Global database manager instance
db = DatabaseManager()
