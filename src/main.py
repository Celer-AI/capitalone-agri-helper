"""Main FastAPI application for Agri-Credit Helper."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
import structlog
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Form, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
        # Test database connection (schema already exists)
        test_result = db.client.table('users').select('count').limit(1).execute()
        logger.info("Database connection verified successfully")
        
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


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Admin dashboard HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agri-Credit Helper Admin</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            .header { text-align: center; margin-bottom: 30px; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
            .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
            .upload-section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            textarea { height: 100px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌾 Agri-Credit Helper Admin Dashboard</h1>
                <p>Document Management & Analytics</p>
            </div>

            <div class="metrics" id="metrics">
                <div class="metric-card">
                    <div class="metric-value" id="total-users">Loading...</div>
                    <div>Total Users</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="total-documents">Loading...</div>
                    <div>Documents</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="daily-queries">Loading...</div>
                    <div>Daily Queries</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="success-rate">Loading...</div>
                    <div>Success Rate</div>
                </div>
            </div>

            <div class="upload-section">
                <h3>📄 Upload Document</h3>
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">PDF File:</label>
                        <input type="file" id="file" name="file" accept=".pdf" required>
                    </div>
                    <div class="form-group">
                        <label for="source-name">Source Name:</label>
                        <input type="text" id="source-name" name="source_name" placeholder="e.g., PM-KISAN Guidelines 2024" required>
                    </div>
                    <button type="submit" class="btn">🚀 Upload & Process</button>
                </form>
                <div id="upload-status"></div>
            </div>

            <div class="upload-section">
                <h3>📝 Add Text Content</h3>
                <form id="text-form">
                    <div class="form-group">
                        <label for="text-source">Source Name:</label>
                        <input type="text" id="text-source" name="source_name" placeholder="e.g., Crop Insurance Policy" required>
                    </div>
                    <div class="form-group">
                        <label for="text-content">Text Content:</label>
                        <textarea id="text-content" name="content" placeholder="Paste policy text here..." required></textarea>
                    </div>
                    <button type="submit" class="btn">🚀 Process Text</button>
                </form>
                <div id="text-status"></div>
            </div>
        </div>

        <script>
            // Load metrics
            async function loadMetrics() {
                try {
                    const response = await fetch('/admin/metrics');
                    const data = await response.json();
                    document.getElementById('total-users').textContent = data.total_users;
                    document.getElementById('total-documents').textContent = data.total_documents;
                    document.getElementById('daily-queries').textContent = data.daily_queries;
                    document.getElementById('success-rate').textContent = data.success_rate.toFixed(1) + '%';
                } catch (error) {
                    console.error('Failed to load metrics:', error);
                }
            }

            // Handle file upload
            document.getElementById('upload-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const statusDiv = document.getElementById('upload-status');

                statusDiv.innerHTML = '<div class="status">Uploading and processing...</div>';

                try {
                    const response = await fetch('/admin/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();

                    if (response.ok) {
                        statusDiv.innerHTML = `<div class="status success">✅ Success! Created ${result.chunks_created} chunks, stored ${result.chunks_stored}</div>`;
                        e.target.reset();
                        loadMetrics();
                    } else {
                        statusDiv.innerHTML = `<div class="status error">❌ Error: ${result.detail}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<div class="status error">❌ Upload failed: ${error.message}</div>`;
                }
            });

            // Handle text upload
            document.getElementById('text-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const statusDiv = document.getElementById('text-status');

                statusDiv.innerHTML = '<div class="status">Processing text...</div>';

                try {
                    const response = await fetch('/admin/text', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();

                    if (response.ok) {
                        statusDiv.innerHTML = `<div class="status success">✅ Success! Created ${result.chunks_created} chunks, stored ${result.chunks_stored}</div>`;
                        e.target.reset();
                        loadMetrics();
                    } else {
                        statusDiv.innerHTML = `<div class="status error">❌ Error: ${result.detail}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<div class="status error">❌ Processing failed: ${error.message}</div>`;
                }
            });

            // Load metrics on page load
            loadMetrics();

            // Refresh metrics every 30 seconds
            setInterval(loadMetrics, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/admin/metrics")
async def get_admin_metrics():
    """Get real-time metrics for admin dashboard."""
    try:
        # Get total users
        try:
            users_result = db.client.table('users').select('count').execute()
            total_users = len(users_result.data) if users_result.data else 0
        except:
            total_users = 0

        # Get total documents
        try:
            docs_result = db.client.table('documents').select('count').execute()
            total_documents = len(docs_result.data) if docs_result.data else 0
        except:
            total_documents = 0

        # Get daily queries
        try:
            from datetime import datetime
            today = datetime.now().date()
            chat_result = db.client.table('chat_history').select('count').gte('created_at', f'{today}T00:00:00').execute()
            daily_queries = len(chat_result.data) if chat_result.data else 0
        except:
            daily_queries = 0

        # Calculate success rate
        try:
            analytics_result = db.client.table('analytics').select('event_type').eq('event_type', 'successful_query').execute()
            successful = len(analytics_result.data) if analytics_result.data else 0
            total_analytics = db.client.table('analytics').select('count').execute()
            total = len(total_analytics.data) if total_analytics.data else 1
            success_rate = (successful / total) * 100 if total > 0 else 0
        except:
            success_rate = 0

        return {
            'total_users': total_users,
            'total_documents': total_documents,
            'daily_queries': daily_queries,
            'success_rate': success_rate
        }
    except Exception as e:
        logger.error("Failed to get admin metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@app.post("/admin/upload")
async def admin_upload_document(file: UploadFile = File(...), source_name: str = Form(...)):
    """Admin endpoint for document upload."""
    try:
        from src.document_processor import document_processor

        # Read file content
        file_content = await file.read()

        # Process the document
        result = await document_processor.process_pdf_file(file_content, source_name)

        if result['success']:
            return {
                "message": "Document processed successfully",
                "filename": result['filename'],
                "chunks_created": result['chunks_created'],
                "chunks_stored": result['chunks_stored']
            }
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Processing failed'))

    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/text")
async def admin_process_text(source_name: str = Form(...), content: str = Form(...)):
    """Admin endpoint for text processing."""
    try:
        from src.document_processor import document_processor

        # Process the text content
        result = await document_processor.process_text_content(content, source_name)

        if result['success']:
            return {
                "message": "Text processed successfully",
                "source": result['filename'],
                "chunks_created": result['chunks_created'],
                "chunks_stored": result['chunks_stored']
            }
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Processing failed'))

    except Exception as e:
        logger.error("Text processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


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
