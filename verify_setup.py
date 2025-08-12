#!/usr/bin/env python3
"""
Quick setup verification script for Agri-Credit Helper
Checks if all required components are properly configured.
"""

import os
import sys
import asyncio
from typing import Dict, List, Tuple

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(filepath)

def check_env_variables() -> Dict[str, bool]:
    """Check if required environment variables are set."""
    required_vars = [
        'GEMINI_API_KEY',
        'TELEGRAM_BOT_TOKEN', 
        'SUPABASE_URL',
        'SUPABASE_SERVICE_ROLE_KEY',
        'COHERE_API_KEY'
    ]
    
    # Load .env file if it exists
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if value and value != 'your_key_here':
                        os.environ[key] = value
    
    return {var: bool(os.getenv(var)) for var in required_vars}

def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 11
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    return is_valid, version_str

def check_dependencies() -> Dict[str, bool]:
    """Check if required Python packages are available."""
    required_packages = [
        'google.genai',
        'supabase', 
        'cohere',
        'telegram',
        'streamlit',
        'fastapi',
        'uvicorn',
        'structlog',
        'numpy',
        'pandas'
    ]
    
    results = {}
    for package in required_packages:
        try:
            __import__(package)
            results[package] = True
        except ImportError:
            results[package] = False
    
    return results

def check_project_structure() -> Dict[str, bool]:
    """Check if project structure is correct."""
    required_files = [
        'src/__init__.py',
        'src/main.py',
        'src/config.py',
        'src/database.py',
        'src/ai_services.py',
        'src/rag_pipeline.py',
        'src/telegram_bot.py',
        'src/document_processor.py',
        'src/admin_dashboard.py',
        'sql/complete_migration.sql',
        'requirements.txt',
        'Dockerfile',
        '.env'
    ]
    
    return {file: check_file_exists(file) for file in required_files}

async def test_basic_imports():
    """Test if core modules can be imported."""
    try:
        sys.path.insert(0, 'src')
        
        from config import settings
        from database import db
        from ai_services import ai_services
        from rag_pipeline import rag_pipeline
        
        return True, "All core modules imported successfully"
    except Exception as e:
        return False, f"Import failed: {str(e)}"

def print_results():
    """Print comprehensive setup verification results."""
    print("🔍 AGRI-CREDIT HELPER SETUP VERIFICATION")
    print("=" * 50)
    
    # Python version
    python_ok, python_version = check_python_version()
    status = "✅" if python_ok else "❌"
    print(f"{status} Python Version: {python_version} {'(OK)' if python_ok else '(Need 3.11+)'}")
    
    # Environment variables
    print("\n📋 Environment Variables:")
    env_vars = check_env_variables()
    for var, status in env_vars.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {var}")
    
    # Project structure
    print("\n📁 Project Structure:")
    structure = check_project_structure()
    missing_files = [f for f, exists in structure.items() if not exists]
    
    if not missing_files:
        print("  ✅ All required files present")
    else:
        print("  ❌ Missing files:")
        for file in missing_files:
            print(f"    - {file}")
    
    # Dependencies
    print("\n📦 Python Dependencies:")
    deps = check_dependencies()
    missing_deps = [pkg for pkg, available in deps.items() if not available]
    
    if not missing_deps:
        print("  ✅ All required packages installed")
    else:
        print("  ❌ Missing packages:")
        for pkg in missing_deps:
            print(f"    - {pkg}")
        print("  💡 Run: pip install -r requirements.txt")
    
    # Module imports
    print("\n🔧 Module Import Test:")
    try:
        import_ok, import_msg = asyncio.run(test_basic_imports())
        icon = "✅" if import_ok else "❌"
        print(f"  {icon} {import_msg}")
    except Exception as e:
        print(f"  ❌ Import test failed: {str(e)}")
        import_ok = False
    
    # Overall status
    print("\n" + "=" * 50)
    
    all_checks = [
        python_ok,
        all(env_vars.values()),
        len(missing_files) == 0,
        len(missing_deps) == 0,
        import_ok
    ]
    
    if all(all_checks):
        print("🎉 SETUP VERIFICATION PASSED!")
        print("✅ Your Agri-Credit Helper is ready for local testing")
        print("\n🚀 Next steps:")
        print("  1. Run: python test_e2e_local.py")
        print("  2. Run: python start_local.py")
        print("  3. Access admin dashboard: http://localhost:8501")
    else:
        print("⚠️ SETUP VERIFICATION FAILED!")
        print("❌ Please fix the issues above before proceeding")
        
        if not python_ok:
            print("\n🐍 Python Issue:")
            print("  - Install Python 3.11 or higher")
        
        if not all(env_vars.values()):
            print("\n🔑 Environment Variables Issue:")
            print("  - Check your .env file")
            print("  - Ensure all API keys are properly set")
        
        if missing_files:
            print("\n📁 Project Structure Issue:")
            print("  - Ensure you're in the correct directory")
            print("  - Re-run the implementation if files are missing")
        
        if missing_deps:
            print("\n📦 Dependencies Issue:")
            print("  - Run: pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    print_results()
