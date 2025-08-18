"""RAG (Retrieve-Augment-Generate) pipeline implementation."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import structlog
from src.ai_services import ai_services, ChatResponse, LanguageDetectionResponse
from src.database import db
from src.config import settings

logger = structlog.get_logger()


class RAGPipeline:
    """Implements the complete RAG pipeline for agricultural finance queries."""
    
    def __init__(self):
        self.ai_services = ai_services
        self.database = db
    
    async def process_query(self, user_query: str, user_id: int, force_language: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Process a user query through the complete RAG pipeline.
        
        Returns:
            Tuple of (response_text, metadata)
        """
        metadata = {
            'original_query': user_query,
            'user_id': user_id,
            'pipeline_steps': {},
            'retrieved_docs': 0,
            'reranked_docs': 0,
            'language_detected': None,
            'success': False
        }
        
        try:
            start_time = time.time()

            # Step 1: Detect language, but allow audio transcription to override it
            final_language = "Hindi" # Default
            if force_language:
                final_language = force_language
                metadata['pipeline_steps']['language_detection'] = 'overridden_by_audio'
            else:
                language_result = await self.ai_services.detect_language(user_query)
                final_language = language_result.detected_language
                metadata['language_confidence'] = language_result.confidence
                metadata['pipeline_steps']['language_detection'] = True

            metadata['language_detected'] = final_language

            # Step 2: Improve query for better retrieval
            improved_query = await self.ai_services.improve_query(user_query)
            metadata['improved_query'] = improved_query
            metadata['pipeline_steps']['query_improvement'] = True

            # Step 3: Generate embedding for the query
            query_embedding = await self.ai_services.generate_embedding(
                improved_query,
                task_type="RETRIEVAL_QUERY"
            )

            if not query_embedding:
                logger.error("Failed to generate query embedding", user_id=user_id)
                return None, metadata

            metadata['pipeline_steps']['embedding_generation'] = True

            # Step 4: Retrieve similar documents
            retrieved_docs = await self.database.similarity_search(
                query_embedding,
                limit=settings.max_retrieval_docs
            )

            metadata['retrieved_docs'] = len(retrieved_docs)
            metadata['pipeline_steps']['document_retrieval'] = True

            if not retrieved_docs:
                logger.warning("No documents retrieved for query", user_id=user_id, query=user_query)
                return await self._generate_no_context_response(user_query, final_language), metadata

            # Step 5: Rerank documents for relevance
            reranked_docs = await self.ai_services.rerank_documents(user_query, retrieved_docs)
            metadata['reranked_docs'] = len(reranked_docs)
            metadata['pipeline_steps']['document_reranking'] = True

            # Step 6: Generate structured response with context
            chat_response = await self.ai_services.generate_response(
                user_query,
                reranked_docs,
                final_language  # Use the final detected language here
            )
            
            if chat_response:
                end_time = time.time()
                response_time_ms = int((end_time - start_time) * 1000)

                metadata['success'] = True
                metadata['pipeline_steps']['response_generation'] = True
                metadata['response_confidence'] = chat_response.confidence_score
                metadata['sources_used'] = chat_response.sources_used
                metadata['response_time_ms'] = response_time_ms

                # Log successful interaction with enhanced metadata
                await self.database.log_analytics_event(
                    'successful_query',
                    user_id=user_id,
                    metadata={
                        'query_length': len(user_query),
                        'retrieved_docs': len(retrieved_docs),
                        'reranked_docs': len(reranked_docs),
                        'language': final_language,
                        'language_confidence': metadata.get('language_confidence', 0.9),
                        'response_length': len(chat_response.response_text),
                        'response_confidence': chat_response.confidence_score,
                        'response_time_ms': response_time_ms
                    }
                )

                # Return the response text for backward compatibility
                return chat_response.response_text, metadata
            else:
                logger.error("Failed to generate response", user_id=user_id)
                return None, metadata
                
        except Exception as e:
            logger.error("RAG pipeline failed", user_id=user_id, error=str(e))
            metadata['error'] = str(e)
            
            # Log failed interaction
            await self.database.log_analytics_event(
                'failed_query',
                user_id=user_id,
                metadata={'error': str(e), 'query': user_query}
            )
            
            return None, metadata
    
    async def _generate_no_context_response(self, query: str, language: str) -> str:
        """Generate a response when no relevant documents are found."""
        fallback_responses = {
            'Hindi': """मुझे खुशी है कि आपने सवाल पूछा है, लेकिन मेरे पास इस विषय पर पूरी जानकारी नहीं है। 

कृपया निम्नलिखित करें:
1. अपने नजदीकी कृषि विभाग के कार्यालय से संपर्क करें
2. किसान कॉल सेंटर (1800-180-1551) पर कॉल करें
3. PM-KISAN या KCC जैसी योजनाओं के लिए अपने बैंक से बात करें

मैं केवल उन सरकारी योजनाओं के बारे में जानकारी दे सकता हूं जो मेरे डेटाबेस में हैं।""",
            
            'Tamil': """உங்கள் கேள்விக்கு நான் மகிழ்ச்சியடைகிறேன், ஆனால் இந்த விषயத்தில் எனக்கு முழுமையான தகவல் இல்லை.

தயவுசெய்து பின்வருவனவற்றைச் செய்யுங்கள்:
1. உங்கள் அருகிலுள்ள வேளாண்மைத் துறை அலுவலகத்தைத் தொடர்பு கொள்ளுங்கள்
2. விவசாயி அழைப்பு மையத்தை (1800-180-1551) அழைக்கவும்
3. PM-KISAN அல்லது KCC போன்ற திட்டங்களுக்கு உங்கள் வங்கியுடன் பேசுங்கள்

எனது தரவுத்தளத்தில் உள்ள அரசாங்க திட்டங்களைப் பற்றி மட்டுமே என்னால் தகவல் வழங்க முடியும்.""",
            
            'English': """I'm glad you asked, but I don't have complete information on this topic.

Please try the following:
1. Contact your nearest Agriculture Department office
2. Call the Kisan Call Center (1800-180-1551)
3. Speak with your bank about schemes like PM-KISAN or KCC

I can only provide information about government schemes that are in my database."""
        }
        
        return fallback_responses.get(language, fallback_responses['English'])
    
    async def process_voice_query(self, audio_data: bytes, mime_type: str, user_id: int) -> Tuple[Optional[str], Dict[str, Any]]:
        """Process voice query through transcription and RAG pipeline."""
        metadata = { 'user_id': user_id, 'input_type': 'voice' }

        try:
            # Step 1: Transcribe audio using the new function
            transcription_result = await self.ai_services.transcribe_audio(audio_data, mime_type)

            if not transcription_result or 'transcribed_english_text' not in transcription_result:
                logger.error("Failed to get valid transcription", user_id=user_id)
                return None, metadata

            transcribed_text = transcription_result['transcribed_english_text']
            detected_language = transcription_result.get('detected_language', 'Hindi') # Default to Hindi

            metadata['transcription'] = transcribed_text
            metadata['detected_language_from_audio'] = detected_language

            # Step 2: Process the ENGLISH transcription through the RAG pipeline
            # We will tell the final generation step to use the original detected language
            response, rag_metadata = await self.process_query(transcribed_text, user_id, force_language=detected_language)

            metadata.update(rag_metadata)
            return response, metadata

        except Exception as e:
            logger.error("Voice query processing failed", user_id=user_id, error=str(e))
            return None, metadata
    
    async def get_conversation_context(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context for a user."""
        try:
            chat_history = await self.database.get_chat_history(user_id, limit * 2)  # Get more to filter
            
            # Format for context
            context = []
            for message in reversed(chat_history):  # Reverse to get chronological order
                context.append({
                    'role': message['role'],
                    'content': message['content'][:500],  # Limit content length
                    'timestamp': message['created_at']
                })
            
            return context[-limit:]  # Return last N messages
            
        except Exception as e:
            logger.error("Failed to get conversation context", user_id=user_id, error=str(e))
            return []
    
    async def store_conversation(self, user_id: int, user_message: str, assistant_response: str,
                               message_type: str = 'text', language_detected: Optional[str] = None,
                               response_time_ms: Optional[int] = None, transcription_data: Optional[Dict] = None):
        """Store conversation in database with enhanced metadata and transcription data."""
        try:
            # Store user message with transcription data if available
            user_chat_id = await self.database.store_chat_message(
                user_id, 'user', user_message,
                message_type=message_type,
                language_detected=language_detected,
                transcription_data=transcription_data
            )

            # Store assistant response
            assistant_chat_id = await self.database.store_chat_message(
                user_id, 'assistant', assistant_response,
                message_type='text',
                language_detected=language_detected,
                response_time_ms=response_time_ms
            )

            logger.info("Conversation stored",
                       user_id=user_id,
                       message_type=message_type,
                       user_chat_id=user_chat_id,
                       assistant_chat_id=assistant_chat_id)

            return user_chat_id, assistant_chat_id

        except Exception as e:
            logger.error("Failed to store conversation", user_id=user_id, error=str(e))
            return None, None


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()
