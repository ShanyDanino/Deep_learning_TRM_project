#!/bin/bash

set -e  # Exit on error

# ============================================================================
# TinyRecursiveModels (TRM) - Complete Training & Evaluation Pipeline
# ============================================================================
# This script provides a one-file solution for building datasets, training,
# and evaluating TRM models on Nonogram tasks.
# ============================================================================

# Detect number of GPUs dynamically
if command -v nvidia-smi &> /dev/null; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    DETECTED_GPUS=0
fi

if [ "$DETECTED_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. This training requires at least 1 GPU."
    echo "Please ensure CUDA and nvidia-smi are properly installed."
    exit 1
fi

# Configuration
SIZE=${1:-5}
SUBSAMPLE_SIZE=${2:-1000}
DATASET_PATH=$3
PROCESSED_DATASET_PATH=$4
NUM_GPUS=$DETECTED_GPUS  # Use all available GPUs

echo "=========================================="
echo "TinyRecursiveModels Training & Evaluation"
echo "=========================================="
echo "Detected GPUs: $DETECTED_GPUS"
echo "Nonogram size: $SIZE X $SIZE"
echo "Subsample size: $SUBSAMPLE_SIZE"
echo "Using GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# ============================================================================
# Step 0: Environment Setup
# ============================================================================

echo "[Step 0/3] Setting up environment..."

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv installed successfully!"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment with uv..."
    #uv venv .venv
fi

# Activate virtual environment
#source .venv/bin/activate

# Install PyTorch with CUDA 12.8 support
echo "Installing PyTorch (CUDA 12.8)..."
uv pip install --pre --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install project dependencies
echo "Installing project dependencies..."
#uv pip install -e .

# Create necessary directories
mkdir -p data checkpoints logs results wandb

# Login to Weights & Biases
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to Weights & Biases..."
    wandb login "$WANDB_API_KEY"
    echo "W&B login successful!"
else
    echo "Weights & Biases API key not found in environment variable WANDB_API_KEY"
    echo "Would you like to enter your W&B API key now? (y/n)"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Enter your W&B API key:"
        read -s wandb_key
        echo "Logging in to Weights & Biases..."
        wandb login "$wandb_key"
        echo "W&B login successful!"
        export WANDB_API_KEY="$wandb_key"
    else
        exit 1
    fi
fi

echo "Environment setup complete!"
echo ""

# ============================================================================
# Step 1: Dataset Building
# ============================================================================

build_nonogram_dataset() {
    echo "[Step 1/3] Building Nonogram dataset..."
    python -u -m src.tiny_recursive_models.data.build_nonogram_dataset \
        --size $SIZE \
        --subsample-size $SUBSAMPLE_SIZE \
		--dataset-path $DATASET_PATH \
		--processed-dataset-path $PROCESSED_DATASET_PATH
    echo "Nonogram with size $SIZE X $SIZE dataset built successfully!"
    echo ""
}

# ============================================================================
# Step 2: Training Functions
# ============================================================================

train_nonogram() {
    local run_name="pretrain_nonogram_$(date +%Y%m%d_%H%M%S)"
    local batch_size=256
    local nproc=$NUM_GPUS
    
    echo "[Step 2/3] Training Nonogram model..."
    echo "Run name: $run_name"
    echo "Batch size: $batch_size"
    echo "GPUs: $nproc"
    echo ""
    
    torchrun --nproc-per-node $nproc --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        scripts/train.py \
        arch=trm \
        data_paths="[data/nonogram_dataset]" \
        evaluators="[]" \
        epochs=50000 eval_interval=5000 \
        lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
        arch.L_layers=2 \
        arch.H_cycles=3 arch.L_cycles=6 \
        lr_warmup_steps=4000 \
        global_batch_size=$batch_size \
        checkpoint_every_eval=True \
        +run_name=${run_name} ema=True
		max_clue_len=
    
    echo "Nonogram training complete!"
    LAST_CHECKPOINT="checkpoints/TRM/${run_name}"
    LAST_DATASET="data/nonogram_dataset"
    echo ""
}

# ============================================================================
# Step 3: Evaluation Function
# ============================================================================

evaluate_model() {
    local checkpoint_path=$1
    local dataset_path=$2
    local eval_name=$(basename $checkpoint_path)
    local nproc=$NUM_GPUS
    local batch_size=256
    
    echo "[Step 3/3] Evaluating model..."
    echo "Checkpoint: $checkpoint_path"
    echo "Dataset: $dataset_path"
    echo "Output directory: checkpoints/eval_${eval_name}"
    echo ""
    
    # Find the latest checkpoint directory
    if [ -d "$checkpoint_path" ]; then
        # Look for the latest checkpoint subdirectory
        local latest_ckpt=$(find "$checkpoint_path" -type d -name "step_*" | sort -V | tail -1)
        
        if [ -z "$latest_ckpt" ]; then
            echo "WARNING: No checkpoint found in $checkpoint_path, skipping evaluation"
            return
        fi
        
        torchrun --nproc-per-node=$nproc scripts/run_eval_only.py \
            --checkpoint "$latest_ckpt" \
            --dataset "$dataset_path" \
            --outdir "checkpoints/eval_${eval_name}" \
            --eval-save-outputs inputs labels puzzle_identifiers preds \
            --global-batch-size $batch_size \
            --apply-ema
        
        echo "Evaluation complete! Results saved to checkpoints/eval_${eval_name}"
    else
        echo "WARNING: Checkpoint path $checkpoint_path does not exist, skipping evaluation"
    fi
    echo ""
}

# ============================================================================
# Main Execution Logic
# IMPORTANT: when re-running you may comment out the building dataset steps
# ============================================================================

build_nonogram_dataset
train_nonogram
evaluate_model "$LAST_CHECKPOINT" "$LAST_DATASET"

echo "  $0 nonogram"

# ============================================================================
# Final Summary
# ============================================================================

echo "=========================================="
echo "Training and Evaluation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  GPUs used: $NUM_GPUS"
if [ -n "$LAST_CHECKPOINT" ]; then
    echo "  Last checkpoint: $LAST_CHECKPOINT"
    echo "  Evaluation results: checkpoints/eval_$(basename $LAST_CHECKPOINT)"
fi
echo ""
echo "Next steps:"
echo "  - Check training logs in: logs/"
echo "  - View checkpoints in: checkpoints/"
echo "  - Review evaluation results in: checkpoints/eval_*/"
if command -v wandb &> /dev/null; then
    echo "  - View training metrics in W&B dashboard"
fi
echo "=========================================="

