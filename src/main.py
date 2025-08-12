"""Main FastAPI application for Agri-Credit Helper."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
import structlog
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import application modules
from src.config import settings
from src.database import db
from src.telegram_bot import telegram_bot
from src.rag_pipeline import rag_pipeline

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Agri-Credit Helper application")
    
    try:
        # Initialize database schema
        await db.initialize_schema()
        logger.info("Database initialized successfully")
        
        # Set up Telegram webhook if URL is provided
        if settings.telegram_webhook_url:
            webhook_success = await telegram_bot.setup_webhook(settings.telegram_webhook_url)
            if webhook_success:
                logger.info("Telegram webhook configured", webhook_url=settings.telegram_webhook_url)
            else:
                logger.warning("Failed to configure Telegram webhook")
        
        # Log startup analytics
        await db.log_analytics_event('application_started', metadata={
            'environment': settings.environment,
            'models': {
                'llm': settings.llm_model,
                'embedding': settings.embedding_model
            }
        })
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agri-Credit Helper application")
    
    # Log shutdown analytics
    await db.log_analytics_event('application_shutdown', metadata={
        'environment': settings.environment
    })


# Create FastAPI application
app = FastAPI(
    title="Agri-Credit Helper API",
    description="AI-powered agricultural finance assistant for Indian farmers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agri-Credit Helper API",
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": structlog.processors.TimeStamper(fmt="iso")(),
            "environment": settings.environment,
            "services": {
                "database": "healthy",  # Could add actual DB ping
                "ai_services": "healthy",  # Could add actual API checks
                "telegram_bot": "healthy"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Telegram webhook."""
    try:
        # Get request data
        update_data = await request.json()
        
        # Process webhook in background
        background_tasks.add_task(
            telegram_bot.handle_webhook,
            update_data
        )
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error("Webhook processing failed", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid webhook data")


@app.post("/query")
async def process_query(request: Request):
    """Direct query endpoint for testing."""
    try:
        data = await request.json()
        query = data.get("query")
        user_id = data.get("user_id", 0)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Process query through RAG pipeline
        response, metadata = await rag_pipeline.process_query(query, user_id)
        
        return {
            "response": response,
            "metadata": metadata,
            "status": "success" if response else "failed"
        }
        
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.get("/stats")
async def get_stats():
    """Get application statistics."""
    try:
        # This would typically query the database for real stats
        stats = {
            "total_users": 0,
            "total_queries": 0,
            "total_documents": 0,
            "success_rate": 0.0,
            "uptime": "unknown"
        }
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@app.post("/admin/upload")
async def admin_upload_document(request: Request):
    """Admin endpoint for document upload."""
    try:
        # This would handle file upload
        # For now, return placeholder
        return {
            "message": "Document upload endpoint",
            "status": "not_implemented"
        }
        
    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(status_code=500, detail="Document upload failed")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error("Unhandled exception", 
                path=request.url.path,
                method=request.method,
                error=str(exc))
    
    # Log error analytics
    await db.log_analytics_event('application_error', metadata={
        'path': request.url.path,
        'method': request.method,
        'error': str(exc)
    })
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.environment == "development"
    )
