#!/bin/bash

# Script to upload PDF documents to the Agri-Credit Helper admin dashboard using curl

set -e

# Configuration
SERVICE_URL="https://agri-credit-helper-whpgm2anha-el.a.run.app"
DOCS_DIR="docs"

echo "🚀 Starting document upload to Agri-Credit Helper"
echo "🌐 Service URL: $SERVICE_URL"
echo ""

# Check service health
echo "🏥 Checking service health..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/health.json "$SERVICE_URL/health")
HEALTH_CODE="${HEALTH_RESPONSE: -3}"

if [ "$HEALTH_CODE" != "200" ]; then
    echo "❌ Service is not healthy (code: $HEALTH_CODE). Aborting."
    exit 1
fi
echo "✅ Service is healthy"

# Get initial metrics
echo ""
echo "📊 Initial metrics:"
curl -s "$SERVICE_URL/admin/metrics" | python3 -m json.tool

echo ""
echo "📁 Uploading PDF documents..."
echo ""

# Function to upload a PDF
upload_pdf() {
    local file_path="$1"
    local source_name="$2"
    local filename=$(basename "$file_path")
    
    echo "📄 Uploading: $filename"
    echo "   Source: $source_name"
    
    if [ ! -f "$file_path" ]; then
        echo "   ⚠️  File not found: $file_path"
        return 1
    fi
    
    # Upload using curl (store response directly without temp file)
    RESPONSE_WITH_CODE=$(curl -s -w "\n%{http_code}" \
        -X POST "$SERVICE_URL/admin/upload" \
        -F "file=@$file_path" \
        -F "source_name=$source_name")
    
    # Extract response code and body
    RESPONSE_CODE=$(echo "$RESPONSE_WITH_CODE" | tail -n1)
    RESPONSE_BODY=$(echo "$RESPONSE_WITH_CODE" | head -n -1)

    if [ "$RESPONSE_CODE" = "200" ]; then
        echo "   ✅ Success!"
        echo "$RESPONSE_BODY" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f'   Created {data.get(\"chunks_created\", 0)} chunks, stored {data.get(\"chunks_stored\", 0)}')
except:
    print('   Upload completed')
"
        return 0
    else
        echo "   ❌ Failed (code: $RESPONSE_CODE)"
        echo "$RESPONSE_BODY"
        return 1
    fi
}

# Upload each PDF document
successful_uploads=0
total_files=0

# Interest Subvention Scheme for KCC
total_files=$((total_files + 1))
if upload_pdf "$DOCS_DIR/Interest Subvention Scheme for KCC.pdf" "Interest Subvention Scheme for Kisan Credit Card"; then
    successful_uploads=$((successful_uploads + 1))
fi
echo ""

# Master Circular - KCC Scheme
total_files=$((total_files + 1))
if upload_pdf "$DOCS_DIR/Master Circular - Kisan Credit Card (KCC) Scheme.pdf" "Master Circular - Kisan Credit Card Scheme"; then
    successful_uploads=$((successful_uploads + 1))
fi
echo ""

# PM-KISAN Scheme Guidelines
total_files=$((total_files + 1))
if upload_pdf "$DOCS_DIR/Revised Operational Guidelines - PM-Kisan Scheme.pdf" "PM-KISAN Scheme Operational Guidelines"; then
    successful_uploads=$((successful_uploads + 1))
fi
echo ""

# PM-KISAN Scheme Guidelines (Revised)
total_files=$((total_files + 1))
if upload_pdf "$DOCS_DIR/Revised Operational Guidelines - PM-Kisan Scheme-2.pdf" "PM-KISAN Scheme Operational Guidelines (Revised)"; then
    successful_uploads=$((successful_uploads + 1))
fi
echo ""

# Revised Operational Guidelines
total_files=$((total_files + 1))
if upload_pdf "$DOCS_DIR/Revised_Operational_Guidelines.pdf" "Revised Operational Guidelines"; then
    successful_uploads=$((successful_uploads + 1))
fi
echo ""

# Get final metrics
echo "📊 Final metrics:"
curl -s "$SERVICE_URL/admin/metrics" | python3 -m json.tool

# Summary
echo ""
echo "=================================================="
echo "📋 UPLOAD SUMMARY"
echo "=================================================="
echo "✅ Successful uploads: $successful_uploads/$total_files"
echo ""

if [ $successful_uploads -gt 0 ]; then
    echo "🎉 Documents uploaded successfully!"
    echo "🔍 You can now test queries like:"
    echo "   - 'What is KCC?'"
    echo "   - 'PM-KISAN scheme eligibility'"
    echo "   - 'How to apply for Kisan Credit Card?'"
    echo ""
    echo "🌐 Admin Dashboard: $SERVICE_URL/admin"
    echo "🧪 Test API:"
    echo "   curl -X POST $SERVICE_URL/query \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"query\": \"What is KCC?\", \"user_id\": 12345}'"
    echo ""
    
    # Test a query
    echo "🧪 Testing query: 'What is KCC?'"
    curl -s -X POST "$SERVICE_URL/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "What is KCC?", "user_id": 12345}' | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Response:', data.get('response', 'No response')[:200] + '...')
print('Retrieved docs:', data.get('metadata', {}).get('retrieved_docs', 0))
print('Success:', data.get('metadata', {}).get('success', False))
"
else
    echo "❌ No documents were uploaded successfully."
fi

# Clean up temp files
rm -f /tmp/health.json /tmp/upload_response.json
