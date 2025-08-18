"""Document processing pipeline for PDF ingestion and text extraction."""

import asyncio
import io
from typing import List, Dict, Any, Optional, Union
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import structlog
from src.ai_services import ai_services
from src.database import db
from src.config import settings

logger = structlog.get_logger()


class DocumentProcessor:
    """Handles document processing, chunking, and storage."""
    
    def __init__(self):
        self.ai_services = ai_services
        self.database = db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def process_pdf_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a PDF file through the complete ingestion pipeline.
        
        Args:
            file_content: PDF file content as bytes
            filename: Name of the source file
            
        Returns:
            Processing result with metadata
        """
        result = {
            'filename': filename,
            'success': False,
            'chunks_created': 0,
            'chunks_stored': 0,
            'error': None,
            'processing_steps': {}
        }
        
        try:
            # Step 1: Upload document to Supabase storage
            storage_url = await self.database.upload_document_to_storage(file_content, filename)
            if not storage_url:
                logger.warning("Failed to upload document to storage, continuing without storage URL", filename=filename)

            result['processing_steps']['storage_upload'] = bool(storage_url)
            result['storage_url'] = storage_url

            # Step 2: Extract text from PDF
            raw_text = await self._extract_text_from_pdf(file_content)
            if not raw_text:
                result['error'] = "Failed to extract text from PDF"
                return result

            result['processing_steps']['text_extraction'] = True
            result['raw_text_length'] = len(raw_text)

            # Step 3: Clean text using AI
            cleaned_text = await self.ai_services.clean_document_text(raw_text)
            if not cleaned_text:
                logger.warning("AI cleaning failed, using raw text", filename=filename)
                cleaned_text = raw_text

            result['processing_steps']['text_cleaning'] = True
            result['cleaned_text_length'] = len(cleaned_text)

            # Step 4: Split text into chunks
            chunks = await self._create_text_chunks(cleaned_text)
            result['chunks_created'] = len(chunks)
            result['processing_steps']['text_chunking'] = True

            if not chunks:
                result['error'] = "No chunks created from text"
                return result

            # Step 5: Generate embeddings for chunks
            chunk_data = await self._generate_chunk_embeddings(chunks, filename, storage_url)
            result['processing_steps']['embedding_generation'] = True

            # Step 6: Store chunks in database
            stored_count = await self._store_chunks(chunk_data)
            result['chunks_stored'] = stored_count
            result['processing_steps']['database_storage'] = True
            
            if stored_count > 0:
                result['success'] = True
                
                # Log successful processing
                await self.database.log_analytics_event(
                    'document_processed',
                    metadata={
                        'filename': filename,
                        'chunks_created': len(chunks),
                        'chunks_stored': stored_count,
                        'raw_text_length': len(raw_text),
                        'cleaned_text_length': len(cleaned_text)
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error("Document processing failed", filename=filename, error=str(e))
            result['error'] = str(e)
            
            # Log failed processing
            await self.database.log_analytics_event(
                'document_processing_failed',
                metadata={'filename': filename, 'error': str(e)}
            )
            
            return result
    
    async def process_text_input(self, text_content: str, source_name: str) -> Dict[str, Any]:
        """
        Process raw text input (bypassing PDF extraction).
        
        Args:
            text_content: Raw text content
            source_name: Name/identifier for the source
            
        Returns:
            Processing result with metadata
        """
        result = {
            'source_name': source_name,
            'success': False,
            'chunks_created': 0,
            'chunks_stored': 0,
            'error': None,
            'processing_steps': {}
        }
        
        try:
            # Step 1: Clean text using AI
            cleaned_text = await self.ai_services.clean_document_text(text_content)
            if not cleaned_text:
                logger.warning("AI cleaning failed, using raw text", source=source_name)
                cleaned_text = text_content
            
            result['processing_steps']['text_cleaning'] = True
            result['cleaned_text_length'] = len(cleaned_text)
            
            # Step 2: Split text into chunks
            chunks = await self._create_text_chunks(cleaned_text)
            result['chunks_created'] = len(chunks)
            result['processing_steps']['text_chunking'] = True
            
            if not chunks:
                result['error'] = "No chunks created from text"
                return result
            
            # Step 3: Generate embeddings for chunks
            chunk_data = await self._generate_chunk_embeddings(chunks, source_name, None)
            result['processing_steps']['embedding_generation'] = True
            
            # Step 4: Store chunks in database
            stored_count = await self._store_chunks(chunk_data)
            result['chunks_stored'] = stored_count
            result['processing_steps']['database_storage'] = True
            
            if stored_count > 0:
                result['success'] = True
                
                # Log successful processing
                await self.database.log_analytics_event(
                    'text_processed',
                    metadata={
                        'source_name': source_name,
                        'chunks_created': len(chunks),
                        'chunks_stored': stored_count,
                        'text_length': len(text_content),
                        'cleaned_text_length': len(cleaned_text)
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error("Text processing failed", source=source_name, error=str(e))
            result['error'] = str(e)
            
            # Log failed processing
            await self.database.log_analytics_event(
                'text_processing_failed',
                metadata={'source_name': source_name, 'error': str(e)}
            )
            
            return result
    
    async def _extract_text_from_pdf(self, file_content: bytes) -> Optional[str]:
        """Extract text from PDF using PyMuPDF."""
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            extracted_text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text with layout information
                text = page.get_text("text")
                if text.strip():
                    extracted_text += f"\n--- Page {page_num + 1} ---\n"
                    extracted_text += text
            
            pdf_document.close()
            
            if extracted_text.strip():
                return extracted_text
            else:
                logger.warning("No text extracted from PDF")
                return None
                
        except Exception as e:
            logger.error("PDF text extraction failed", error=str(e))
            return None
    
    async def _create_text_chunks(self, text: str) -> List[str]:
        """Split text into chunks using LangChain text splitter."""
        try:
            chunks = self.text_splitter.split_text(text)
            
            # Filter out very short chunks
            filtered_chunks = [
                chunk.strip() for chunk in chunks 
                if len(chunk.strip()) > 50  # Minimum chunk length
            ]
            
            logger.info("Text chunking completed", 
                       original_chunks=len(chunks), 
                       filtered_chunks=len(filtered_chunks))
            
            return filtered_chunks
            
        except Exception as e:
            logger.error("Text chunking failed", error=str(e))
            return []
    
    async def _generate_chunk_embeddings(self, chunks: List[str], source_document: str, storage_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks."""
        try:
            # Generate embeddings in batches
            batch_size = 10
            chunk_data = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Generate embeddings for batch
                embeddings = await self.ai_services.generate_embeddings_batch(
                    batch_chunks, 
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                # Combine chunks with embeddings
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    if embedding:  # Only include chunks with successful embeddings
                        chunk_data.append({
                            'content': chunk,
                            'embedding': embedding,
                            'source_document': source_document,
                            'storage_url': storage_url
                        })
                    else:
                        logger.warning("Failed to generate embedding for chunk", 
                                     chunk_index=i + j, 
                                     source=source_document)
                
                # Small delay between batches to avoid rate limiting
                await asyncio.sleep(0.1)
            
            logger.info("Embedding generation completed", 
                       total_chunks=len(chunks), 
                       successful_embeddings=len(chunk_data))
            
            return chunk_data
            
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            return []
    
    async def _store_chunks(self, chunk_data: List[Dict[str, Any]]) -> int:
        """Store chunks with embeddings in database."""
        try:
            success = await self.database.store_document_chunks(chunk_data)
            
            if success:
                return len(chunk_data)
            else:
                return 0
                
        except Exception as e:
            logger.error("Chunk storage failed", error=str(e))
            return 0
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        try:
            # This would require additional database queries
            # For now, return basic stats
            return {
                'total_documents': 0,  # Would query from analytics
                'total_chunks': 0,     # Would query from documents table
                'processing_success_rate': 0.0
            }
            
        except Exception as e:
            logger.error("Failed to get processing stats", error=str(e))
            return {}


# Global document processor instance
document_processor = DocumentProcessor()
