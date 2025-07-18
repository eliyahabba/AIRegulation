#!/bin/bash

#SBATCH --mem=32g
#SBATCH --time=18:0:0
#SBATCH --gres=gpu:h200:1
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --killable
#SBATCH --exclude=drape-01,drape-02,drape-03

# Usage: sbatch ./run_dataset_on_model_3_8b.sh [model_name] [quantization]
# Examples:
#   sbatch ./run_dataset_on_model_3_8b.sh                    # Uses llama3_8b with 8bit quantization
#   sbatch ./run_dataset_on_model_3_8b.sh llama3_1b          # Uses llama3_1b with 8bit quantization
#   sbatch ./run_dataset_on_model_3_8b.sh llama3_8b 4bit     # Uses llama3_8b with 4bit quantization
#   sbatch ./run_dataset_on_model_3_8b.sh phi_3_mini none    # Uses phi_3_mini without quantization

# Set Hugging Face cache directory
export HF_HOME="/cs/snapless/gabis/gabis/shared"
echo "HF_HOME is set to: $HF_HOME"

# Set up project directory and Python path
project_dir="../../"
absolute_project_path=$(readlink -f $project_dir)
export PYTHONPATH=$absolute_project_path:$PYTHONPATH
echo "PYTHONPATH is set to: $PYTHONPATH"

# Show job information (only if running on SLURM)
if [ -n "$SLURM_JOB_ID" ]; then
    sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
fi

# Load required modules (only if available)
if command -v module &> /dev/null; then
    module load cuda
    module load torch
fi

# Change to project directory
echo "Current dir is set to: $absolute_project_path"
cd $absolute_project_path

# Set environment variables
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

# Get model name from command line argument
MODEL_NAME=${1:-llama3_8b}  # Default to llama3_8b if no argument provided
QUANTIZATION=${2:-8bit}     # Default to 8bit if no argument provided

# Validate model name
case $MODEL_NAME in
    llama3_1b|llama3_3b|llama3_8b|phi_3_mini|phi_4_mini|qwen_1_5b|qwen_7b|qwen3_8b|mistral_7b|mistral_8b|gemma_7b|gemma_2_9b|gemma_3_12b|deepseek_7b|vicuna_7b|falcon_7b|mpt_7b|olmo_2_13b|mixtral_8x7b|dolly_v2_7b|yi_34b|aya_expanse_8b|command_r7b)
        echo "✅ Using model: $MODEL_NAME"
        ;;
    *)
        echo "❌ Error: Invalid model name '$MODEL_NAME'"
        echo "Available models: llama3_1b, llama3_3b, llama3_8b, phi_3_mini, phi_4_mini, qwen_1_5b, qwen_7b, qwen3_8b, mistral_7b, mistral_8b, gemma_7b, gemma_2_9b, gemma_3_12b, deepseek_7b, vicuna_7b, falcon_7b, mpt_7b, olmo_2_13b, mixtral_8x7b, dolly_v2_7b, yi_34b, aya_expanse_8b, command_r7b"
        exit 1
        ;;
esac

# Validate quantization
case $QUANTIZATION in
    8bit|4bit|none)
        echo "✅ Using quantization: $QUANTIZATION"
        ;;
    *)
        echo "❌ Error: Invalid quantization '$QUANTIZATION'"
        echo "Available quantization: 8bit, 4bit, none"
        exit 1
        ;;
esac

# Run AIR-Bench batch processing
# Parameters:
# --platform local: Use local model
# --model $MODEL_NAME: Use the specified model
# --quantization $QUANTIZATION: Use specified quantization
# --parallel_workers 1: Use sequential processing (1 worker)
# --all: Process all available AIR-Bench categories
# --max_tokens 100: Maximum tokens for response
# --temperature 0.0: Deterministic responses
# --batch_size 50: Process 50 variations before saving
# --max_retries 3: Retry failed requests up to 3 times
# --retry_sleep 60: Wait 60 seconds between retries

echo "Starting AIR-Bench batch processing with model: $MODEL_NAME (quantization: $QUANTIZATION)..."
python src/execution/run_airbench_batch.py \
    --platform local \
    --model $MODEL_NAME \
    --quantization $QUANTIZATION \
    --parallel_workers 1 \
    --all \
    --max_tokens 1000 \
    --temperature 0.0 \
    --batch_size 50 \
    --max_retries 3 \
    --retry_sleep 60

echo "AIR-Bench batch processing completed!"
