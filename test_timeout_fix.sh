#!/bin/bash
# Quick test script to verify the timeout fix

echo "=========================================="
echo "Testing EEG Upload with Timeout Fix"
echo "=========================================="
echo ""

# Configuration
API_URL="http://localhost:3000/api/analyze-eeg"
TEST_FILE="../sample/0001082.edf"
REPO_ROOT="/home/zik/UniStuff/FYP/stESI_pub"

cd "$REPO_ROOT/frontend" || exit 1

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

echo "📁 Test file: $TEST_FILE"
echo "🌐 API endpoint: $API_URL"
echo "⏱️  Expected time: 30-60 seconds (100 windows)"
echo ""
echo "Starting upload..."
echo ""

# Record start time
START_TIME=$(date +%s)

# Make the API request
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
  -F "file=@$TEST_FILE" \
  --max-time 300)

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Extract HTTP status code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | head -n -1)

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo ""
echo "⏱️  Processing time: ${DURATION}s"
echo "🔢 HTTP status: $HTTP_CODE"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ SUCCESS! EEG processing completed"
    echo ""
    echo "Response preview:"
    echo "$BODY" | jq '.message, .processingTime, .animationFile' 2>/dev/null || echo "$BODY" | head -c 200
    echo ""
    echo "=========================================="
    echo "Test PASSED ✅"
    echo "=========================================="
    exit 0
else
    echo "❌ FAILED with status $HTTP_CODE"
    echo ""
    echo "Error details:"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    echo ""
    echo "=========================================="
    echo "Test FAILED ❌"
    echo "=========================================="
    exit 1
fi
