"""Tests for document processing functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test cases for document processor."""
    
    @pytest.fixture
    def document_processor(self):
        """Create document processor instance for testing."""
        return DocumentProcessor()
    
    @pytest.mark.asyncio
    async def test_process_text_input_success(self, document_processor):
        """Test successful text input processing."""
        test_text = "This is a test government policy document with important information about agricultural schemes."
        
        with patch.object(document_processor.ai_services, 'clean_document_text', return_value='cleaned text'), \
             patch.object(document_processor, '_create_text_chunks', return_value=['chunk1', 'chunk2']), \
             patch.object(document_processor, '_generate_chunk_embeddings', return_value=[
                 {'content': 'chunk1', 'embedding': [0.1] * 768, 'source_document': 'test'},
                 {'content': 'chunk2', 'embedding': [0.2] * 768, 'source_document': 'test'}
             ]), \
             patch.object(document_processor, '_store_chunks', return_value=2), \
             patch.object(document_processor.database, 'log_analytics_event', return_value=True):
            
            result = await document_processor.process_text_input(test_text, "test_source")
            
            assert result['success'] is True
            assert result['chunks_created'] == 2
            assert result['chunks_stored'] == 2
    
    @pytest.mark.asyncio
    async def test_create_text_chunks(self, document_processor):
        """Test text chunking functionality."""
        long_text = "This is a long text. " * 100  # Create text longer than chunk size
        
        chunks = await document_processor._create_text_chunks(long_text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(len(chunk) > 50 for chunk in chunks)  # All chunks should be above minimum length
    
    @pytest.mark.asyncio
    async def test_generate_chunk_embeddings(self, document_processor):
        """Test embedding generation for chunks."""
        chunks = ['chunk1', 'chunk2', 'chunk3']
        
        with patch.object(document_processor.ai_services, 'generate_embeddings_batch', return_value=[
            [0.1] * 768, [0.2] * 768, [0.3] * 768
        ]):
            
            chunk_data = await document_processor._generate_chunk_embeddings(chunks, "test_source")
            
            assert len(chunk_data) == 3
            assert all('content' in item for item in chunk_data)
            assert all('embedding' in item for item in chunk_data)
            assert all('source_document' in item for item in chunk_data)
    
    @pytest.mark.asyncio
    async def test_process_pdf_file_extraction_failure(self, document_processor):
        """Test PDF processing when text extraction fails."""
        fake_pdf_data = b"fake pdf content"
        
        with patch.object(document_processor, '_extract_text_from_pdf', return_value=None):
            
            result = await document_processor.process_pdf_file(fake_pdf_data, "test.pdf")
            
            assert result['success'] is False
            assert 'Failed to extract text from PDF' in result['error']


if __name__ == "__main__":
    pytest.main([__file__])
