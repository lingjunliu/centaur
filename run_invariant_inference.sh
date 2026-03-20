#!/bin/bash

# Script to run invariant inference for all APIs in the problematic APIs list

API_LIST_FILE="${1:-apis_list.txt}"
LOG_FILE="invariant_inference_run.log"

# Check if API list file exists
if [ ! -f "$API_LIST_FILE" ]; then
    echo "Error: API list file not found: $API_LIST_FILE"
    exit 1
fi

# Count total APIs
TOTAL=$(wc -l < "$API_LIST_FILE")

echo "========================================"
echo "Running Invariant Inference"
echo "========================================"
echo "API List: $API_LIST_FILE"
echo "Total APIs: $TOTAL"
echo "Log File: $LOG_FILE"
echo "Started at: $(date)"
echo "========================================"
echo ""

# Initialize log file
echo "Invariant Inference Run Log" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "Total APIs: $TOTAL" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Initialize counters
SUCCESS=0
FAILED=0
COUNTER=0

# Read each API and run the command
while IFS= read -r API; do
    # Skip empty lines
    [ -z "$API" ] && continue

    COUNTER=$((COUNTER + 1))

    echo "[$COUNTER/$TOTAL] Processing: $API"
    echo "" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"
    echo "[$COUNTER/$TOTAL] API: $API" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"

    # Run the command
    echo "  Running: python3 -m learner.invariant_inference $API 300 1 tf 0"

    if python3 -m learner.invariant_inference "$API" 300 1 tf 0 >> "$LOG_FILE" 2>&1; then
        echo "  ✓ SUCCESS"
        echo "Status: SUCCESS" >> "$LOG_FILE"
        SUCCESS=$((SUCCESS + 1))
    else
        EXIT_CODE=$?
        echo "  ✗ FAILED (exit code: $EXIT_CODE)"
        echo "Status: FAILED (exit code: $EXIT_CODE)" >> "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

done < "$API_LIST_FILE"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total APIs processed: $COUNTER"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "Finished at: $(date)"
echo "========================================"
echo "Detailed log saved to: $LOG_FILE"

# Write summary to log
echo "" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "SUMMARY" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Total APIs processed: $COUNTER" >> "$LOG_FILE"
echo "Successful: $SUCCESS" >> "$LOG_FILE"
echo "Failed: $FAILED" >> "$LOG_FILE"
echo "Finished at: $(date)" >> "$LOG_FILE"
