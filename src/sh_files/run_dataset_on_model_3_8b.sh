#!/bin/bash

#SBATCH --mem=16g
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:a5000:1
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

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

# Run AIR-Bench batch processing
# Parameters:
# --platform local: Use local model
# --model llama3_8b: Use Llama 3.1 8B model
# --quantization 8bit: Use 8-bit quantization for memory efficiency
# --parallel_workers 1: Use sequential processing (1 worker)
# --all: Process all available AIR-Bench categories
# --max_tokens 1024: Maximum tokens for response
# --temperature 0.0: Deterministic responses
# --batch_size 50: Process 50 variations before saving
# --max_retries 3: Retry failed requests up to 3 times
# --retry_sleep 60: Wait 60 seconds between retries

echo "Starting AIR-Bench batch processing with Llama 3.1 8B model..."
python src/execution/run_airbench_batch.py \
    --platform local \
    --model llama3_8b \
    --quantization 8bit \
    --parallel_workers 1 \
    --all \
    --max_tokens 100 \
    --temperature 0.0 \
    --batch_size 50 \
    --max_retries 3 \
    --retry_sleep 60

echo "AIR-Bench batch processing completed!"
