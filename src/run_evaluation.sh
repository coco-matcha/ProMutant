#!/bin/bash
set -e

# Configuration
API_KEY="${OPENAI_API_KEY:-your-api-key-here}"
MODEL="${MODEL:-gpt-4o}"
EXPERIMENT_ID="${EXPERIMENT_ID:-bt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-back_translation_flipped}"
NUM_LOOPS="${NUM_LOOPS:-2}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
SKIP_COUNT="${SKIP_COUNT:-0}"

# Derived paths
PRED_DIR="experiments/${EXPERIMENT_NAME}/split_predictions"
BASE_NAME="${MODEL//-/_}_predictions"
RUN_ID="${EXPERIMENT_ID}_${MODEL//-/_}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_step "Checking Dependencies"
    
    if ! command_exists python3; then
        print_error "python3 is required but not installed"
        exit 1
    fi
    
    # Check if swebench is available
    if ! python3 -c "import swebench" 2>/dev/null; then
        print_warning "swebench not found. Install with: pip install swebench"
    fi
    
    print_success "Dependencies check completed"
}

# Run the back translation pipeline
run_pipeline() {
    print_step "Running Back Translation Pipeline"
    
    # Build command arguments
    local cmd_args="--api-key \"$API_KEY\" --model \"$MODEL\" --experiment-id \"$EXPERIMENT_ID\" --experiment-name \"$EXPERIMENT_NAME\" --num-loops $NUM_LOOPS --max-retries $MAX_RETRIES --skip-count $SKIP_COUNT"
    
    if [ -n "$SAMPLE_SIZE" ]; then
        cmd_args="$cmd_args --sample-size $SAMPLE_SIZE"
    fi
    
    echo "Running: python3 run_back_translation_pipeline.py $cmd_args"
    
    if eval "python3 run_back_translation_pipeline.py $cmd_args"; then
        print_success "Back translation pipeline completed successfully"
    else
        print_error "Back translation pipeline failed"
        exit 1
    fi
}

# Count the number of prediction files
count_prediction_files() {
    if [ ! -d "$PRED_DIR" ]; then
        print_error "Prediction directory not found: $PRED_DIR"
        exit 1
    fi
    
    local count=$(find "$PRED_DIR" -name "${BASE_NAME}_*.jsonl" | wc -l)
    echo "$count"
}

# Run evaluations on split files
run_evaluations() {
    print_step "Running SWE-Bench Evaluations"
    
    local num_files=$(count_prediction_files)
    
    if [ "$num_files" -eq 0 ]; then
        print_error "No prediction files found in $PRED_DIR"
        exit 1
    fi
    
    print_success "Found $num_files prediction files to evaluate"
    
    local completed=0
    local failed=0
    
    for i in $(seq 1 "$num_files"); do
        local pred_file="${PRED_DIR}/${BASE_NAME}_${i}.jsonl"
        local current_run_id="${RUN_ID}.${i}"
        
        if [ ! -f "$pred_file" ]; then
            print_warning "Skipping missing file: $pred_file"
            continue
        fi
        
        echo ""
        print_step "Evaluation $i/$num_files"
        echo "File: $pred_file"
        echo "Run ID: $current_run_id"
        
        if python3 -m swebench.harness.run_evaluation \
            --dataset_name princeton-nlp/SWE-bench_Lite \
            --predictions_path "$pred_file" \
            --max_workers 8 \
            --run_id "$current_run_id"; then
            
            completed=$((completed + 1))
            print_success "Completed evaluation $i/$num_files"
        else
            failed=$((failed + 1))
            print_error "Failed evaluation $i/$num_files"
        fi
        
        echo "Progress: $completed completed, $failed failed"
        echo "----------------------------------------"
    done
    
    echo ""
    if [ "$failed" -eq 0 ]; then
        print_success "All $completed evaluations completed successfully!"
    else
        print_warning "$completed evaluations completed, $failed failed"
    fi
}

# Resume from existing predictions (skip pipeline)
resume_evaluations() {
    print_step "Resuming from Existing Predictions"
    
    if [ ! -d "$PRED_DIR" ]; then
        print_error "Prediction directory not found: $PRED_DIR"
        print_error "Cannot resume - run the full pipeline first"
        exit 1
    fi
    
    run_evaluations
}

# Show usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -r, --resume            Resume from existing predictions (skip pipeline)"
    echo "  -k, --api-key KEY       OpenAI API key (or set OPENAI_API_KEY env var)"
    echo "  -m, --model MODEL       Model to use (default: gpt-4o)"
    echo "  -e, --experiment-id ID  Experiment identifier (default: bt)"
    echo "  -n, --num-loops N       Number of back-translation loops (default: 2)"
    echo "  -s, --sample-size N     Sample size from dataset (default: use all)"
    echo "  --skip-count N          Number of items to skip (default: 0)"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY         OpenAI API key"
    echo "  MODEL                  Model to use"
    echo "  EXPERIMENT_ID          Experiment identifier"
    echo "  NUM_LOOPS              Number of loops"
    echo "  SAMPLE_SIZE            Sample size"
    echo "  SKIP_COUNT             Items to skip"
    echo ""
    echo "Examples:"
    echo "  $0 --api-key sk-... --model gpt-4"
    echo "  $0 --resume"
    echo "  $0 --sample-size 50 --num-loops 3"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -r|--resume)
            RESUME=true
            shift
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--experiment-id)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        -n|--num-loops)
            NUM_LOOPS="$2"
            shift 2
            ;;
        -s|--sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --skip-count)
            SKIP_COUNT="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "Back Translation Evaluation Script"
    echo "================================="
    echo "Configuration:"
    echo "  Model: $MODEL"
    echo "  Experiment ID: $EXPERIMENT_ID" 
    echo "  Num Loops: $NUM_LOOPS"
    echo "  Sample Size: ${SAMPLE_SIZE:-all}"
    echo "  Skip Count: $SKIP_COUNT"
    echo "  Resume: ${RESUME:-false}"
    echo ""
    
    # Check if API key is provided
    if [ "$API_KEY" = "your-api-key-here" ] || [ -z "$API_KEY" ]; then
        print_error "Please provide an OpenAI API key via --api-key or OPENAI_API_KEY environment variable"
        exit 1
    fi
    
    check_dependencies
    
    if [ "${RESUME:-false}" = "true" ]; then
        resume_evaluations
    else
        run_pipeline
        run_evaluations
    fi
    
    print_success "All operations completed!"
}

# Run main function
main