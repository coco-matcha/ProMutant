#!/bin/bash
set -e

# Configuration
EXPERIMENT_ID="promutant"
MODEL_NAME="gpt-4o"
EXPERIMENTS_DIR="data/promutant"
PRED_DIR="$EXPERIMENTS_DIR/split_predictions"
BASE_NAME="${MODEL_NAME}_predictions"
API_KEY="${OPENAI_API_KEY}"  # Get from environment

# Validate API key
if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

# 1. Query LLM to generate mutants
echo "Step 1: Generating mutants..."
python3 src/generate_mutants.py \
    --api-key "$API_KEY" \
    --model "$MODEL_NAME" \
    --experiment-name "$EXPERIMENT_ID" \
    --output-dir "$EXPERIMENTS_DIR"

# 2. Create prediction format
echo "Step 2: Creating prediction format..."
python3 src/create_prediction_format.py \
    --mutants-file "$EXPERIMENTS_DIR/${EXPERIMENT_ID}_${MODEL_NAME}_output.jsonl" \
    --output-dir "$EXPERIMENTS_DIR" \
    --model-name "$MODEL_NAME" \
    --split-prefix "${MODEL_NAME}_predictions"

# Count files matching pattern
NUM_FILES=$(find "$PRED_DIR" -name "${BASE_NAME}_*.jsonl" -type f | wc -l)
echo "Found $NUM_FILES prediction files"

# Check if files exist
if [ "$NUM_FILES" -eq 0 ]; then
    echo "No prediction files found in $PRED_DIR"
    exit 1
fi

# 3. Evaluation using SWE-Bench harness
echo "Step 3: Running evaluations..."
for i in $(seq 1 "$NUM_FILES"); do
    pred_file="${PRED_DIR}/${BASE_NAME}_${i}.jsonl"
    
    # Check if file exists before processing
    if [ ! -f "$pred_file" ]; then
        echo "Warning: File $pred_file does not exist, skipping"
        continue
    fi
    
    echo "Running evaluation for ${BASE_NAME}_${i}.jsonl with run_id ${EXPERIMENT_ID}.${i}"
    
    python3 -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Lite \
        --predictions_path "$pred_file" \
        --max_workers 8 \
        --run_id "${EXPERIMENT_ID}.${i}"
    
    echo "Completed evaluation $i/$NUM_FILES"
    echo "----------------------------------------"
done

echo "All evaluations completed!"

# 4. Calculate coupling metrics
echo "Step 4: Calculating metrics..."

# Copy metrics.py to current directory temporarily
cp src/evaluation/metrics.py ./metrics_temp.py

# Run metrics calculation
echo -e "$MODEL_NAME\n$EXPERIMENT_ID.1-$NUM_FILES" | python3 src/evaluation/metrics.py

# Clean up temporary file
rm metrics_temp.py