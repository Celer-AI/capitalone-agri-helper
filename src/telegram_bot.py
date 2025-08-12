"""Telegram bot integration with webhook support."""

import asyncio
import io
from typing import Optional, Dict, Any
import httpx
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, CommandHandler, filters
import structlog
from src.rag_pipeline import rag_pipeline
from src.database import db
from src.config import settings

logger = structlog.get_logger()


class TelegramBotHandler:
    """Handles Telegram bot interactions with rate limiting and analytics."""
    
    def __init__(self):
        self.bot = Bot(token=settings.telegram_bot_token)
        self.rag_pipeline = rag_pipeline
        self.database = db
    
    async def setup_webhook(self, webhook_url: str) -> bool:
        """Set up webhook for the Telegram bot."""
        try:
            await self.bot.set_webhook(url=webhook_url)
            logger.info("Webhook set successfully", webhook_url=webhook_url)
            return True
        except Exception as e:
            logger.error("Failed to set webhook", webhook_url=webhook_url, error=str(e))
            return False
    
    async def handle_webhook(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook data from Telegram."""
        try:
            update = Update.de_json(update_data, self.bot)
            
            if update and update.message:
                await self._process_message(update.message)
                return {"status": "success"}
            
            return {"status": "no_message"}
            
        except Exception as e:
            logger.error("Failed to handle webhook", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def _process_message(self, message):
        """Process incoming message from user."""
        user_id = message.from_user.id
        
        try:
            # Check rate limiting
            if not await self._check_rate_limit(user_id):
                await self._send_rate_limit_message(message.chat_id)
                return
            
            # Log analytics
            await self.database.log_analytics_event(
                'message_received',
                user_id=user_id,
                metadata={
                    'message_type': 'voice' if message.voice else 'text',
                    'chat_id': message.chat_id
                }
            )
            
            # Process based on message type
            if message.voice:
                await self._handle_voice_message(message)
            elif message.text:
                await self._handle_text_message(message)
            else:
                await self._send_unsupported_message(message.chat_id)
                
        except Exception as e:
            logger.error("Failed to process message", user_id=user_id, error=str(e))
            await self._send_error_message(message.chat_id)
    
    async def _handle_text_message(self, message):
        """Handle text message from user."""
        user_id = message.from_user.id
        user_text = message.text
        
        try:
            # Handle commands
            if user_text.startswith('/'):
                await self._handle_command(message)
                return
            
            # Send typing indicator
            await self.bot.send_chat_action(chat_id=message.chat_id, action="typing")
            
            # Process through RAG pipeline
            response, metadata = await self.rag_pipeline.process_query(user_text, user_id)
            
            if response:
                # Increment user chat count
                await self.database.increment_user_chat_count(user_id)

                # Store conversation with enhanced metadata
                await self.rag_pipeline.store_conversation(
                    user_id,
                    user_text,
                    response,
                    message_type='text',
                    language_detected=metadata.get('language_detected'),
                    response_time_ms=metadata.get('response_time_ms')
                )

                # Send response
                await self.bot.send_message(
                    chat_id=message.chat_id,
                    text=response,
                    parse_mode='Markdown'
                )

                # Send feedback buttons
                await self._send_feedback_buttons(message.chat_id)
                
            else:
                await self._send_processing_error_message(message.chat_id)
                
        except Exception as e:
            logger.error("Failed to handle text message", user_id=user_id, error=str(e))
            await self._send_error_message(message.chat_id)
    
    async def _handle_voice_message(self, message):
        """Handle voice message from user."""
        user_id = message.from_user.id
        
        try:
            # Send typing indicator
            await self.bot.send_chat_action(chat_id=message.chat_id, action="typing")
            
            # Download voice file
            voice_file = await self.bot.get_file(message.voice.file_id)
            
            # Download audio data
            async with httpx.AsyncClient() as client:
                response = await client.get(voice_file.file_path)
                audio_data = response.content
            
            # Process through RAG pipeline
            response, metadata = await self.rag_pipeline.process_voice_query(
                audio_data, 
                "audio/ogg",  # Telegram voice messages are in OGG format
                user_id
            )
            
            if response:
                # Increment user chat count
                await self.database.increment_user_chat_count(user_id)

                # Store conversation with enhanced metadata (use transcription as user message)
                transcription = metadata.get('transcription', '[Voice Message]')
                await self.rag_pipeline.store_conversation(
                    user_id,
                    transcription,
                    response,
                    message_type='voice',
                    language_detected=metadata.get('language_detected'),
                    response_time_ms=metadata.get('response_time_ms')
                )

                # Send response
                await self.bot.send_message(
                    chat_id=message.chat_id,
                    text=response,
                    parse_mode='Markdown'
                )

                # Send feedback buttons
                await self._send_feedback_buttons(message.chat_id)
                
            else:
                await self._send_voice_processing_error_message(message.chat_id)
                
        except Exception as e:
            logger.error("Failed to handle voice message", user_id=user_id, error=str(e))
            await self._send_error_message(message.chat_id)
    
    async def _handle_command(self, message):
        """Handle bot commands."""
        command = message.text.lower()
        
        if command == '/start':
            await self._send_welcome_message(message.chat_id)
        elif command == '/help':
            await self._send_help_message(message.chat_id)
        elif command == '/status':
            await self._send_status_message(message.chat_id, message.from_user.id)
        else:
            await self._send_unknown_command_message(message.chat_id)
    
    async def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user has exceeded daily chat limit."""
        try:
            current_count = await self.database.get_user_chat_count(user_id)
            return current_count < settings.daily_chat_limit
        except Exception as e:
            logger.error("Failed to check rate limit", user_id=user_id, error=str(e))
            return True  # Allow on error
    
    async def _send_welcome_message(self, chat_id: int):
        """Send welcome message to user."""
        welcome_text = """🌾 नमस्ते! मैं Agri-Credit Helper हूं।

मैं भारतीय किसानों को सरकारी कृषि वित्त योजनाओं के बारे में जानकारी देने में मदद करता हूं।

आप मुझसे पूछ सकते हैं:
• किसान क्रेडिट कार्ड (KCC) के बारे में
• PM-KISAN योजना के बारे में  
• कृषि लोन और ब्याज दरों के बारे में
• सरकारी सब्सिडी के बारे में

आप हिंदी, तमिल, या अंग्रेजी में टेक्स्ट या वॉइस मैसेज भेज सकते हैं।

/help - अधिक जानकारी के लिए"""
        
        await self.bot.send_message(chat_id=chat_id, text=welcome_text)
    
    async def _send_help_message(self, chat_id: int):
        """Send help message to user."""
        help_text = """📚 मदद - Agri-Credit Helper

🎯 मैं क्या कर सकता हूं:
• सरकारी कृषि योजनाओं की जानकारी
• लोन और क्रेडिट की जानकारी
• सब्सिडी और वित्तीय सहायता की जानकारी

💬 कैसे इस्तेमाल करें:
• सीधे अपना सवाल टाइप करें
• वॉइस मैसेज भेजें
• हिंदी, तमिल, या अंग्रेजी में पूछें

⚡ दैनिक सीमा: 30 सवाल प्रति दिन

📞 अधिक मदद के लिए:
• किसान कॉल सेंटर: 1800-180-1551
• अपने नजदीकी कृषि विभाग से संपर्क करें"""
        
        await self.bot.send_message(chat_id=chat_id, text=help_text)
    
    async def _send_status_message(self, chat_id: int, user_id: int):
        """Send user status message."""
        try:
            current_count = await self.database.get_user_chat_count(user_id)
            remaining = max(0, settings.daily_chat_limit - current_count)
            
            status_text = f"""📊 आपकी स्थिति:

आज के सवाल: {current_count}/{settings.daily_chat_limit}
बचे हुए सवाल: {remaining}

कल फिर से {settings.daily_chat_limit} सवाल पूछ सकेंगे।"""
            
            await self.bot.send_message(chat_id=chat_id, text=status_text)
            
        except Exception as e:
            logger.error("Failed to send status message", user_id=user_id, error=str(e))
    
    async def _send_rate_limit_message(self, chat_id: int):
        """Send rate limit exceeded message."""
        limit_text = f"""⚠️ दैनिक सीमा पूरी हो गई

आपने आज {settings.daily_chat_limit} सवाल पूछ लिए हैं।
कल फिर से सवाल पूछ सकेंगे।

तत्काल मदद के लिए:
📞 किसान कॉल सेंटर: 1800-180-1551"""
        
        await self.bot.send_message(chat_id=chat_id, text=limit_text)
    
    async def _send_feedback_buttons(self, chat_id: int):
        """Send feedback buttons after response."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        keyboard = [
            [
                InlineKeyboardButton("👍 मददगार", callback_data="feedback_helpful"),
                InlineKeyboardButton("👎 मददगार नहीं", callback_data="feedback_not_helpful")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await self.bot.send_message(
            chat_id=chat_id,
            text="क्या यह जानकारी मददगार थी?",
            reply_markup=reply_markup
        )
    
    async def _send_error_message(self, chat_id: int):
        """Send generic error message."""
        error_text = """😔 क्षमा करें, कुछ तकनीकी समस्या हुई है।

कृपया:
• कुछ देर बाद फिर कोशिश करें
• या किसान कॉल सेंटर पर कॉल करें: 1800-180-1551"""
        
        await self.bot.send_message(chat_id=chat_id, text=error_text)
    
    async def _send_processing_error_message(self, chat_id: int):
        """Send processing error message."""
        error_text = """😔 मुझे आपके सवाल का जवाब नहीं मिल सका।

कृपया:
• अपना सवाल दूसरे तरीके से पूछें
• या किसान कॉल सेंटर पर कॉल करें: 1800-180-1551"""
        
        await self.bot.send_message(chat_id=chat_id, text=error_text)
    
    async def _send_voice_processing_error_message(self, chat_id: int):
        """Send voice processing error message."""
        error_text = """🎤 आपका वॉइस मैसेज समझ नहीं आया।

कृपया:
• साफ आवाज में दोबारा रिकॉर्ड करें
• या टेक्स्ट मैसेज भेजें"""
        
        await self.bot.send_message(chat_id=chat_id, text=error_text)
    
    async def _send_unsupported_message(self, chat_id: int):
        """Send unsupported message type error."""
        unsupported_text = """📝 केवल टेक्स्ट और वॉइस मैसेज भेजें।

अन्य फाइल टाइप समर्थित नहीं हैं।"""
        
        await self.bot.send_message(chat_id=chat_id, text=unsupported_text)
    
    async def _send_unknown_command_message(self, chat_id: int):
        """Send unknown command message."""
        unknown_text = """❓ अज्ञात कमांड।

उपलब्ध कमांड:
/start - शुरुआत करें
/help - मदद
/status - आपकी स्थिति"""
        
        await self.bot.send_message(chat_id=chat_id, text=unknown_text)


# Global bot handler instance
telegram_bot = TelegramBotHandler()
