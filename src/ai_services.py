"""AI services for embedding, generation, and reranking."""

import asyncio
import time
from typing import List, Dict, Any, Optional
import numpy as np
from google import genai
from google.genai import types
import cohere
import structlog
from asyncio_throttle import Throttler
from src.config import settings, SYSTEM_PROMPT, DOCUMENT_CLEANING_PROMPT
from pydantic import BaseModel
from typing import Literal

logger = structlog.get_logger()


# Structured output models for AI responses
class LanguageDetectionResponse(BaseModel):
    """Structured response for language detection."""
    detected_language: Literal["Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia", "English"]
    confidence: float
    reasoning: str

class QueryImprovementResponse(BaseModel):
    """Structured response for query improvement."""
    improved_query: str
    original_query: str
    improvements_made: List[str]
    search_keywords: List[str]

class ChatResponse(BaseModel):
    """Structured response for chat interactions."""
    response_text: str
    language: str
    confidence_score: float
    sources_used: List[str]
    response_type: Literal["direct_answer", "no_context", "partial_info"]
    suggested_actions: List[str]


class AIServices:
    """Manages all AI-related operations with rate limiting."""

    def __init__(self):
        # Initialize Google GenAI client with API key
        self.genai_client = genai.Client(api_key=settings.gemini_api_key)

        # Initialize Cohere client (using ClientV2 for latest API)
        self.cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)

        # Rate limiting for Gemini Embedding API (100 RPM, 30k TPM)
        self.embedding_throttler = Throttler(rate_limit=90, period=60)  # 90 per minute to be safe

        # Track token usage for TPM limiting
        self.token_usage_window = []
        self.max_tokens_per_minute = settings.gemini_embedding_tpm_limit
    
    async def _check_token_limit(self, estimated_tokens: int) -> bool:
        """Check if we're within token per minute limits."""
        current_time = time.time()

        # Remove tokens older than 1 minute
        self.token_usage_window = [
            (tokens, timestamp) for tokens, timestamp in self.token_usage_window
            if current_time - timestamp < 60
        ]

        # Calculate current usage
        current_usage = sum(tokens for tokens, _ in self.token_usage_window)

        # Check if adding new tokens would exceed limit
        if current_usage + estimated_tokens > self.max_tokens_per_minute:
            logger.warning("Token limit would be exceeded",
                         current_usage=current_usage,
                         estimated_tokens=estimated_tokens,
                         limit=self.max_tokens_per_minute)
            return False

        return True

    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> Optional[List[float]]:
        """Generate embedding for text using Gemini embedding model with rate limiting."""
        try:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(text) // 4

            # Check token limits
            if not await self._check_token_limit(estimated_tokens):
                await asyncio.sleep(10)  # Wait before retrying
                if not await self._check_token_limit(estimated_tokens):
                    logger.error("Token limit exceeded, skipping embedding generation")
                    return None

            # Apply rate limiting
            async with self.embedding_throttler:
                result = self.genai_client.models.embed_content(
                    model=settings.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=settings.embedding_dimensions
                    )
                )

                # Track token usage
                self.token_usage_window.append((estimated_tokens, time.time()))

            if result.embeddings:
                embedding_values = result.embeddings[0].values

                # Normalize embedding for better similarity search
                if settings.embedding_dimensions != 3072:
                    embedding_array = np.array(embedding_values)
                    normalized_embedding = embedding_array / np.linalg.norm(embedding_array)
                    return normalized_embedding.tolist()

                return embedding_values

            return None

        except Exception as e:
            logger.error("Failed to generate embedding", text_length=len(text), error=str(e))
            return None
    
    async def generate_embeddings_batch(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        try:
            result = self.genai_client.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimensions
                )
            )
            
            embeddings = []
            for embedding_obj in result.embeddings:
                if embedding_obj and embedding_obj.values:
                    embedding_values = embedding_obj.values
                    
                    # Normalize if not using default dimension
                    if settings.embedding_dimensions != 3072:
                        embedding_array = np.array(embedding_values)
                        normalized_embedding = embedding_array / np.linalg.norm(embedding_array)
                        embeddings.append(normalized_embedding.tolist())
                    else:
                        embeddings.append(embedding_values)
                else:
                    embeddings.append(None)
            
            return embeddings
            
        except Exception as e:
            logger.error("Failed to generate batch embeddings", batch_size=len(texts), error=str(e))
            return [None] * len(texts)
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/ogg") -> Optional[Dict[str, Any]]:
        """Transcribe audio using Gemini with structured output including language detection."""
        try:
            # Create audio part from bytes
            audio_part = types.Part.from_bytes(
                data=audio_data,
                mime_type=mime_type
            )

            prompt = """Transcribe this audio message and provide structured output. The speaker is likely an Indian farmer asking about agricultural finance or government schemes.

Respond with this exact JSON structure:
{
    "transcribed_text": "The exact transcription in the original language",
    "detected_language": "Hindi|Tamil|Telugu|Bengali|Marathi|Gujarati|Kannada|Malayalam|Punjabi|Odia|English",
    "confidence": 0.95,
    "intent_category": "scheme_inquiry|loan_question|eligibility_check|general_agriculture|unclear",
    "key_terms": ["PM-KISAN", "loan", "subsidy"]
}

Rules:
1. Transcribe exactly what was said, preserving the original language
2. Detect the primary language used
3. Identify the intent category based on content
4. Extract key agricultural/financial terms mentioned
5. Set confidence based on audio clarity and language detection certainty"""

            response = self.genai_client.models.generate_content(
                model=settings.llm_model,
                contents=[prompt, audio_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            if response.text:
                try:
                    import json
                    return json.loads(response.text)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse transcription JSON, returning text only", error=str(e))
                    return {
                        "transcribed_text": response.text,
                        "detected_language": "Unknown",
                        "confidence": 0.7,
                        "intent_category": "unclear",
                        "key_terms": []
                    }

            return None

        except Exception as e:
            logger.error("Failed to transcribe audio", error=str(e))
            return None
    
    async def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using Cohere's rerank API v3.5 with proper error handling."""
        try:
            if not documents:
                logger.warning("No documents provided for reranking")
                return []

            # Clean and validate query
            if not query or not query.strip():
                logger.warning("Empty query provided for reranking")
                return documents[:settings.rerank_top_k]

            query = query.strip()

            # Prepare documents for reranking with better context management
            doc_texts = []
            doc_mapping = []
            total_tokens = 0
            max_tokens = 4000  # Conservative limit for rerank-v3.5

            for i, doc in enumerate(documents):
                content = doc.get('content', '').strip()
                if not content:
                    continue

                # Estimate tokens (rough: 1 token ≈ 4 chars)
                content_tokens = len(content) // 4

                # Check if adding this document would exceed limit
                if total_tokens + content_tokens > max_tokens:
                    logger.info("Rerank token limit reached",
                               docs_included=len(doc_texts),
                               total_docs=len(documents),
                               tokens_used=total_tokens)
                    break

                doc_texts.append(content)
                doc_mapping.append(i)
                total_tokens += content_tokens

            if not doc_texts:
                logger.warning("No valid documents after filtering")
                return documents[:settings.rerank_top_k]

            # Call Cohere rerank API with proper error handling
            logger.info("Calling Cohere rerank",
                       query_length=len(query),
                       doc_count=len(doc_texts),
                       estimated_tokens=total_tokens)

            rerank_response = self.cohere_client.rerank(
                model="rerank-v3.5",  # Use explicit model name
                query=query,
                documents=doc_texts,
                top_n=min(settings.rerank_top_k, len(doc_texts)),
                return_documents=False
            )

            if not rerank_response or not hasattr(rerank_response, 'results'):
                logger.error("Invalid rerank response")
                return documents[:settings.rerank_top_k]

            # Process results
            reranked_docs = []
            for result in rerank_response.results:
                if result.index >= len(doc_mapping):
                    logger.warning("Invalid result index", index=result.index, max_index=len(doc_mapping)-1)
                    continue

                original_index = doc_mapping[result.index]
                original_doc = documents[original_index]
                reranked_docs.append({
                    **original_doc,
                    'rerank_score': result.relevance_score
                })

            logger.info("Rerank completed successfully",
                       input_docs=len(documents),
                       processed_docs=len(doc_texts),
                       output_docs=len(reranked_docs))

            return reranked_docs if reranked_docs else documents[:settings.rerank_top_k]

        except Exception as e:
            logger.error("Rerank failed with error",
                        error=str(e),
                        error_type=type(e).__name__,
                        query_length=len(query) if query else 0,
                        doc_count=len(documents))
            # Always return something
            return documents[:settings.rerank_top_k]
    
    async def generate_response(self, query: str, context_docs: List[Dict[str, Any]], user_language: str = "Hindi") -> Optional[ChatResponse]:
        """Generate structured response using Gemini with context documents."""
        try:
            # Check if this is a greeting
            greeting_words = ['hi', 'hello', 'hey', 'namaste', 'namaskar', 'hola', 'helo', 'hii', 'helllo']
            query_lower = query.lower().strip()

            if any(greeting in query_lower for greeting in greeting_words) and len(query_lower.split()) <= 3:
                # This is a greeting, return welcome message
                welcome_message = {
                    'Hindi': """नमस्ते! मैं आपका कृषि वित्त सहायक हूँ। मैं भारतीय किसानों की सरकारी योजनाओं, ऋण और सब्सिडी के बारे में जानकारी देता हूँ।

आप मुझसे पूछ सकते हैं:
🌾 किसान क्रेडिट कार्ड (KCC) के बारे में
💰 PM-KISAN योजना की जानकारी
🏦 कृषि ऋण की पात्रता
📋 आवेदन प्रक्रिया और दस्तावेज
📞 स्थानीय कार्यालयों की जानकारी

कृपया अपना प्रश्न पूछें!""",
                    'English': """Hello! I'm your Agricultural Finance Assistant. I help Indian farmers with information about government schemes, loans, and subsidies.

You can ask me about:
🌾 Kisan Credit Card (KCC)
💰 PM-KISAN scheme information
🏦 Agricultural loan eligibility
📋 Application processes and documents
📞 Local office information

Please ask your question!""",
                    'Tamil': """வணக்கம்! நான் உங்கள் விவசாய நிதி உதவியாளர். இந்திய விவசாயிகளுக்கு அரசு திட்டங்கள், கடன்கள் மற்றும் மானியங்கள் பற்றிய தகவல்களை வழங்குகிறேன்.

நீங்கள் என்னிடம் கேட்கலாம்:
🌾 கிசான் கிரெடிட் கார்டு (KCC)
💰 PM-KISAN திட்ட தகவல்
🏦 விவசாய கடன் தகுதி
📋 விண்ணப்ப செயல்முறைகள்
📞 உள்ளூர் அலுவலக தகவல்

தயவுசெய்து உங்கள் கேள்வியைக் கேளுங்கள்!"""
                }

                response_text = welcome_message.get(user_language, welcome_message['Hindi'])

                return ChatResponse(
                    response_text=response_text,
                    language=user_language,
                    confidence_score=1.0,
                    sources_used=[],
                    response_type="direct_answer",
                    suggested_actions=["Ask about KCC", "Ask about PM-KISAN", "Ask about loan eligibility"]
                )

            # Prepare context from documents
            context_text = "\n\n".join([
                f"Document: {doc['source_document']}\nContent: {doc['content']}"
                for doc in context_docs
            ])

            sources_list = [doc.get('source_document', 'Unknown') for doc in context_docs]

            # Construct prompt that handles both context and general knowledge
            if context_docs and len(context_text.strip()) > 100:
                # We have good context documents
                prompt = f"""Based on the following government policy documents, provide a structured JSON response about agricultural finance schemes.

CONTEXT DOCUMENTS:
{context_text}

USER QUESTION: {query}

If the context documents contain relevant information, use them. If not, you may use your general knowledge about Indian agricultural schemes but MUST add a disclaimer.

Respond with this exact JSON structure:
{{
    "response_text": "Your helpful answer in {user_language}",
    "language": "{user_language}",
    "confidence_score": 0.95,
    "sources_used": {sources_list},
    "response_type": "direct_answer",
    "suggested_actions": ["Contact local agriculture office", "Apply for scheme X"]
}}

Rules:
1. response_text MUST be in {user_language}
2. If using context documents: confidence_score 0.9+, sources_used from documents
3. If using general knowledge: Add disclaimer "⚠️ यह जानकारी सामान्य ज्ञान पर आधारित है, कृपया आधिकारिक स्रोतों से पुष्टि करें" (in Hindi) or equivalent in other languages
4. confidence_score: 0.7 for general knowledge, sources_used: ["General Knowledge"]
5. suggested_actions: practical next steps for the farmer"""
            else:
                # No relevant context, use general knowledge with disclaimer
                prompt = f"""You are an expert on Indian agricultural finance and government schemes. Answer this question using your general knowledge.

USER QUESTION: {query}

Respond with this exact JSON structure:
{{
    "response_text": "Your helpful answer in {user_language} with disclaimer",
    "language": "{user_language}",
    "confidence_score": 0.7,
    "sources_used": ["General Knowledge"],
    "response_type": "general_knowledge",
    "suggested_actions": ["Contact local agriculture office", "Visit official government websites"]
}}

Rules:
1. response_text MUST be in {user_language}
2. MUST include disclaimer: "⚠️ यह जानकारी सामान्य ज्ञान पर आधारित है, कृपया आधिकारिक स्रोतों से पुष्टि करें" (in Hindi) or equivalent
3. Provide accurate information about Indian agricultural schemes, KCC, PM-KISAN, etc.
4. Focus on practical, actionable advice for Indian farmers"""

            response = self.genai_client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            if response.text:
                try:
                    import json
                    response_data = json.loads(response.text)
                    return ChatResponse(**response_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse structured response, creating fallback", error=str(e))
                    # Fallback to simple response
                    return ChatResponse(
                        response_text=response.text,
                        language=user_language,
                        confidence_score=0.7,
                        sources_used=sources_list,
                        response_type="direct_answer",
                        suggested_actions=["Contact local agriculture office for more details"]
                    )

            return None

        except Exception as e:
            logger.error("Failed to generate response", query_length=len(query), context_docs_count=len(context_docs), error=str(e))
            return None
    
    async def clean_document_text(self, raw_text: str) -> Optional[str]:
        """Clean document text using AI."""
        try:
            response = self.genai_client.models.generate_content(
                model=settings.llm_model,
                contents=f"{DOCUMENT_CLEANING_PROMPT}\n\n{raw_text}",
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )
            
            return response.text if response.text else None
            
        except Exception as e:
            logger.error("Failed to clean document text", text_length=len(raw_text), error=str(e))
            return None
    
    async def detect_language(self, text: str) -> LanguageDetectionResponse:
        """Detect the language of user input with structured output."""
        try:
            prompt = f"""Analyze the language of this text and respond with JSON:

Text: {text}

Respond with this exact JSON structure:
{{
    "detected_language": "Hindi|Tamil|Telugu|Bengali|Marathi|Gujarati|Kannada|Malayalam|Punjabi|Odia|English",
    "confidence": 0.95,
    "reasoning": "Brief explanation of detection"
}}

Rules:
1. detected_language must be one of the listed options
2. confidence: 0.9+ if very sure, 0.7-0.8 if somewhat sure, 0.5-0.6 if uncertain
3. reasoning: brief explanation of why you detected this language"""

            response = self.genai_client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            if response.text:
                try:
                    import json
                    response_data = json.loads(response.text)
                    return LanguageDetectionResponse(**response_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse language detection response", error=str(e))

            # Fallback
            return LanguageDetectionResponse(
                detected_language="Hindi",
                confidence=0.5,
                reasoning="Fallback due to parsing error"
            )

        except Exception as e:
            logger.error("Failed to detect language", text_length=len(text), error=str(e))
            return LanguageDetectionResponse(
                detected_language="Hindi",
                confidence=0.5,
                reasoning="Error in detection process"
            )
    
    async def improve_query(self, user_query: str) -> str:
        """Improve user query for better retrieval."""
        try:
            response = self.genai_client.models.generate_content(
                model=settings.llm_model,
                contents=f"""You are helping to improve a farmer's query for searching agricultural finance information. 

Original query: {user_query}

Please rewrite this query to be more specific and effective for searching government agricultural finance schemes and policies. Focus on:
1. Key terms related to agricultural finance
2. Specific scheme names if mentioned
3. Clear intent about what information is needed

Respond with only the improved query, nothing else.""",
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )
            
            improved_query = response.text.strip() if response.text else user_query
            return improved_query
            
        except Exception as e:
            logger.error("Failed to improve query", query=user_query, error=str(e))
            return user_query  # Return original query on failure


# Global AI services instance
ai_services = AIServices()
