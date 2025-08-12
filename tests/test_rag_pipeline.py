"""Tests for RAG pipeline functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    """Test cases for RAG pipeline."""
    
    @pytest.fixture
    def rag_pipeline(self):
        """Create RAG pipeline instance for testing."""
        return RAGPipeline()
    
    @pytest.mark.asyncio
    async def test_process_query_success(self, rag_pipeline):
        """Test successful query processing."""
        # Mock dependencies
        with patch.object(rag_pipeline.ai_services, 'detect_language', return_value='Hindi'), \
             patch.object(rag_pipeline.ai_services, 'improve_query', return_value='improved query'), \
             patch.object(rag_pipeline.ai_services, 'generate_embedding', return_value=[0.1] * 768), \
             patch.object(rag_pipeline.database, 'similarity_search', return_value=[{'content': 'test doc', 'source_document': 'test.pdf'}]), \
             patch.object(rag_pipeline.ai_services, 'rerank_documents', return_value=[{'content': 'test doc', 'source_document': 'test.pdf'}]), \
             patch.object(rag_pipeline.ai_services, 'generate_response', return_value='Test response'), \
             patch.object(rag_pipeline.database, 'log_analytics_event', return_value=True):
            
            response, metadata = await rag_pipeline.process_query("test query", 12345)
            
            assert response == 'Test response'
            assert metadata['success'] is True
            assert metadata['language_detected'] == 'Hindi'
    
    @pytest.mark.asyncio
    async def test_process_query_no_documents(self, rag_pipeline):
        """Test query processing when no documents are found."""
        with patch.object(rag_pipeline.ai_services, 'detect_language', return_value='Hindi'), \
             patch.object(rag_pipeline.ai_services, 'improve_query', return_value='improved query'), \
             patch.object(rag_pipeline.ai_services, 'generate_embedding', return_value=[0.1] * 768), \
             patch.object(rag_pipeline.database, 'similarity_search', return_value=[]):
            
            response, metadata = await rag_pipeline.process_query("test query", 12345)
            
            assert response is not None  # Should return fallback response
            assert metadata['retrieved_docs'] == 0
    
    @pytest.mark.asyncio
    async def test_process_voice_query(self, rag_pipeline):
        """Test voice query processing."""
        audio_data = b"fake audio data"
        
        with patch.object(rag_pipeline.ai_services, 'transcribe_audio', return_value='transcribed text'), \
             patch.object(rag_pipeline, 'process_query', return_value=('response', {'success': True})):
            
            response, metadata = await rag_pipeline.process_voice_query(audio_data, "audio/ogg", 12345)
            
            assert response == 'response'
            assert metadata['transcription'] == 'transcribed text'
            assert metadata['input_type'] == 'voice'
    
    @pytest.mark.asyncio
    async def test_store_conversation(self, rag_pipeline):
        """Test conversation storage."""
        with patch.object(rag_pipeline.database, 'store_chat_message', return_value=True):
            await rag_pipeline.store_conversation(12345, "user message", "assistant response")
            
            # Verify both messages were stored
            assert rag_pipeline.database.store_chat_message.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
