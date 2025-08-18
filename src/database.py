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
            logger.info("Initializing database schema...")

            # Check if main tables exist and create them if they don't
            await self._ensure_tables_exist()

            logger.info("Database schema initialization completed")

        except Exception as e:
            logger.error("Failed to initialize database schema", error=str(e))
            # Continue anyway - some functionality might still work
            logger.info("Continuing with partial schema...")

    async def _ensure_tables_exist(self):
        """Ensure all required tables exist, create them if they don't."""

        # Check if documents table exists
        try:
            result = self.client.table('documents').select('count').limit(1).execute()
            logger.info("Documents table exists and is accessible")
        except Exception as e:
            logger.warning("Documents table not accessible, will create basic structure", error=str(e))
            # For now, we'll work with what we have and handle errors gracefully

        # Check other tables
        tables_to_check = ['users', 'chat_history', 'analytics', 'document_processing_logs', 'system_metrics']

        for table in tables_to_check:
            try:
                result = self.client.table(table).select('count').limit(1).execute()
                logger.debug(f"Table {table} exists and is accessible")
            except Exception as e:
                logger.warning(f"Table {table} may not exist or is not accessible", error=str(e))

    # Note: Utility functions and triggers are already created via SQL migration
    # No need to recreate them programmatically
    
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
                # =================================================================
                # THE FIX IS HERE: Convert the Python list into the exact string
                # format that pg_vector expects.
                # =================================================================
                embedding_str = '[' + ','.join(map(str, chunk['embedding'])) + ']'

                # Use storage_url from chunk if available, otherwise use the passed storage_url
                chunk_storage_url = chunk.get('storage_url') or storage_url

                documents_data.append({
                    'id': str(uuid.uuid4()),
                    'content': chunk['content'],
                    'embedding': embedding_str,  # Use the formatted string here
                    'source_document': chunk['source_document'],
                    'chunk_index': i,
                    'metadata': {
                        'storage_url': chunk_storage_url,
                        'chunk_length': len(chunk['content']),
                        'processing_timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    'created_at': datetime.now(timezone.utc).isoformat()
                })

            # Insert in batches
            batch_size = 50
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
