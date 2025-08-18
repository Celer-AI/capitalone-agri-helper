#!/usr/bin/env python3
"""
Script to upload PDF documents to the Agri-Credit Helper admin dashboard.
"""

import os
import requests
import json
from pathlib import Path

# Configuration
SERVICE_URL = "https://agri-credit-helper-whpgm2anha-el.a.run.app"
DOCS_DIR = "docs"

# PDF files to upload
PDF_FILES = [
    {
        "file": "Interest Subvention Scheme for KCC.pdf",
        "source_name": "Interest Subvention Scheme for Kisan Credit Card"
    },
    {
        "file": "Master Circular - Kisan Credit Card (KCC) Scheme.pdf", 
        "source_name": "Master Circular - Kisan Credit Card Scheme"
    },
    {
        "file": "Revised Operational Guidelines - PM-Kisan Scheme.pdf",
        "source_name": "PM-KISAN Scheme Operational Guidelines"
    },
    {
        "file": "Revised Operational Guidelines - PM-Kisan Scheme-2.pdf",
        "source_name": "PM-KISAN Scheme Operational Guidelines (Revised)"
    },
    {
        "file": "Revised_Operational_Guidelines.pdf",
        "source_name": "Revised Operational Guidelines"
    }
]

def upload_pdf(file_path: str, source_name: str) -> dict:
    """Upload a PDF file to the admin dashboard."""
    url = f"{SERVICE_URL}/admin/upload"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            data = {'source_name': source_name}
            
            print(f"📄 Uploading: {os.path.basename(file_path)}")
            print(f"   Source: {source_name}")
            
            response = requests.post(url, files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success! Created {result.get('chunks_created', 0)} chunks, stored {result.get('chunks_stored', 0)}")
                return {"success": True, "result": result}
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {"success": False, "error": str(e)}

def get_metrics() -> dict:
    """Get current metrics from the admin dashboard."""
    url = f"{SERVICE_URL}/admin/metrics"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get metrics: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main function to upload all documents."""
    print("🚀 Starting document upload to Agri-Credit Helper")
    print(f"🌐 Service URL: {SERVICE_URL}")
    print()
    
    # Check service health
    try:
        health_response = requests.get(f"{SERVICE_URL}/health", timeout=30)
        if health_response.status_code != 200:
            print("❌ Service is not healthy. Aborting.")
            return
        print("✅ Service is healthy")
    except Exception as e:
        print(f"❌ Cannot reach service: {e}")
        return
    
    # Get initial metrics
    print("\n📊 Initial metrics:")
    initial_metrics = get_metrics()
    if "error" not in initial_metrics:
        print(f"   Documents: {initial_metrics.get('total_documents', 0)}")
        print(f"   Users: {initial_metrics.get('total_users', 0)}")
    
    print(f"\n📁 Found {len(PDF_FILES)} PDF files to upload")
    print()
    
    # Upload each PDF
    successful_uploads = 0
    total_chunks_created = 0
    total_chunks_stored = 0
    
    for pdf_info in PDF_FILES:
        file_path = os.path.join(DOCS_DIR, pdf_info["file"])
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        result = upload_pdf(file_path, pdf_info["source_name"])
        
        if result["success"]:
            successful_uploads += 1
            if "result" in result:
                total_chunks_created += result["result"].get("chunks_created", 0)
                total_chunks_stored += result["result"].get("chunks_stored", 0)
        
        print()  # Empty line for readability
    
    # Get final metrics
    print("📊 Final metrics:")
    final_metrics = get_metrics()
    if "error" not in final_metrics:
        print(f"   Documents: {final_metrics.get('total_documents', 0)}")
        print(f"   Users: {final_metrics.get('total_users', 0)}")
    
    # Summary
    print("\n" + "="*50)
    print("📋 UPLOAD SUMMARY")
    print("="*50)
    print(f"✅ Successful uploads: {successful_uploads}/{len(PDF_FILES)}")
    print(f"📄 Total chunks created: {total_chunks_created}")
    print(f"💾 Total chunks stored: {total_chunks_stored}")
    print()
    
    if successful_uploads > 0:
        print("🎉 Documents uploaded successfully!")
        print("🔍 You can now test queries like:")
        print("   - 'What is KCC?'")
        print("   - 'PM-KISAN scheme eligibility'")
        print("   - 'How to apply for Kisan Credit Card?'")
        print()
        print(f"🌐 Admin Dashboard: {SERVICE_URL}/admin")
        print(f"🧪 Test API: curl -X POST {SERVICE_URL}/query -H 'Content-Type: application/json' -d '{{\"query\": \"What is KCC?\", \"user_id\": 12345}}'")
    else:
        print("❌ No documents were uploaded successfully.")

if __name__ == "__main__":
    main()
