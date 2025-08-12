#!/usr/bin/env python3
"""
Local Development Startup Script for Agri-Credit Helper
This script starts all services for local development and testing.
"""

import asyncio
import subprocess
import sys
import os
import time
import signal
from typing import List
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class LocalDevelopmentServer:
    """Manages local development services."""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True
    
    def start_main_api(self):
        """Start the main FastAPI application."""
        logger.info("🚀 Starting main API server...")
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8080",
            "--log-level", "debug"
        ])
        self.processes.append(process)
        return process
    
    def start_admin_dashboard(self):
        """Start the Streamlit admin dashboard."""
        logger.info("📊 Starting admin dashboard...")
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", 
            "run", "src/admin_dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ])
        self.processes.append(process)
        return process
    
    async def run_initial_tests(self):
        """Run initial system tests."""
        logger.info("🧪 Running initial system tests...")
        try:
            process = subprocess.run([
                sys.executable, "test_e2e_local.py"
            ], capture_output=True, text=True, timeout=120)
            
            if process.returncode == 0:
                logger.info("✅ Initial tests passed!")
                print(process.stdout)
            else:
                logger.error("❌ Initial tests failed!")
                print(process.stderr)
                return False
        except subprocess.TimeoutExpired:
            logger.error("⏰ Tests timed out")
            return False
        except Exception as e:
            logger.error("🚨 Test execution failed", error=str(e))
            return False
        
        return True
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("🛑 Received shutdown signal, stopping services...")
            self.running = False
            self.stop_all_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def stop_all_services(self):
        """Stop all running services."""
        logger.info("🔄 Stopping all services...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.error("Failed to stop process", error=str(e))
        
        self.processes.clear()
        logger.info("✅ All services stopped")
    
    async def start_all_services(self):
        """Start all development services."""
        logger.info("🌟 Starting Agri-Credit Helper Local Development Environment")
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Check environment
        if not os.path.exists('.env'):
            logger.error("❌ .env file not found! Please create it from .env.example")
            return False
        
        # Run initial tests
        logger.info("🔍 Running system checks...")
        tests_passed = await self.run_initial_tests()
        
        if not tests_passed:
            logger.error("❌ System checks failed. Please fix issues before starting services.")
            return False
        
        # Start services
        try:
            # Start main API
            api_process = self.start_main_api()
            
            # Wait a bit for API to start
            await asyncio.sleep(3)
            
            # Start admin dashboard
            dashboard_process = self.start_admin_dashboard()
            
            # Wait a bit for dashboard to start
            await asyncio.sleep(3)
            
            # Display startup information
            self.display_startup_info()
            
            # Keep services running
            await self.monitor_services()
            
        except Exception as e:
            logger.error("Failed to start services", error=str(e))
            self.stop_all_services()
            return False
        
        return True
    
    def display_startup_info(self):
        """Display startup information."""
        print("\n" + "="*60)
        print("🌾 AGRI-CREDIT HELPER - LOCAL DEVELOPMENT")
        print("="*60)
        print("🚀 Services Started Successfully!")
        print("")
        print("📡 Main API Server:")
        print("   URL: http://localhost:8080")
        print("   Health: http://localhost:8080/health")
        print("   Docs: http://localhost:8080/docs")
        print("")
        print("📊 Admin Dashboard:")
        print("   URL: http://localhost:8501")
        print("   Password: AgriCredit2024!")
        print("")
        print("🧪 Testing:")
        print("   Run tests: python test_e2e_local.py")
        print("   Test query: curl -X POST http://localhost:8080/query \\")
        print("              -H 'Content-Type: application/json' \\")
        print("              -d '{\"query\": \"What is KCC?\", \"user_id\": 12345}'")
        print("")
        print("📱 Telegram Bot:")
        print("   Bot: @capital_schemesbot")
        print("   Webhook: http://localhost:8080/webhook")
        print("   Note: Use ngrok for external webhook testing")
        print("")
        print("🛑 Stop Services: Ctrl+C")
        print("="*60)
    
    async def monitor_services(self):
        """Monitor running services."""
        logger.info("👀 Monitoring services... Press Ctrl+C to stop")
        
        while self.running:
            # Check if processes are still running
            for i, process in enumerate(self.processes):
                if process.poll() is not None:
                    logger.error(f"Process {i} has stopped unexpectedly")
                    self.running = False
                    break
            
            await asyncio.sleep(5)
        
        self.stop_all_services()


async def main():
    """Main entry point."""
    server = LocalDevelopmentServer()
    
    try:
        success = await server.start_all_services()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
