#!/bin/bash

# Create output directories
mkdir -p eval_out/full

# Define models
MODELS=(
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_sft"
    "DeepSeek-R1-Distill-Llama-8B-STAR_mix2_safekey"
)

# Define common evaluation parameters
RANDOM_SEED=46
DEBUG=False

# Function to run evaluation for a specific benchmark
run_eval() {
    local benchmark=$1
    local model=$2
    local gpu_ids=$3
    local max_tokens=$4
    local ngpu=$5
    local output_dir=$6

    echo "Running ${benchmark} evaluation for model: ${model}"
    CUDA_VISIBLE_DEVICES=${gpu_ids} python simple_evals.py \
        --model="${model}" \
        --eval_mode="${benchmark}" \
        --debug ${DEBUG} \
        --random_seed=${RANDOM_SEED} \
        --ngpu ${ngpu} \
        --max_tokens ${max_tokens} \
        > ${output_dir}/${benchmark}_${model##*/}.out 2>&1
}

echo "Starting evaluations for all benchmarks with random seed ${RANDOM_SEED}..."

# human_eval evaluations
HumanEval evaluations
echo "Running HumanEval evaluations..."
run_eval "humaneval" "${MODELS[0]}" "4,5,6,7" 16384 4 "eval_out/full"
run_eval "humaneval" "${MODELS[1]}" "4,5,6,7" 16384 4 "eval_out/full"

# MATH evaluations
echo "Running MATH evaluations..."
run_eval "math" "${MODELS[0]}" "4,5,6,7" 16384 4 "eval_out/full"
run_eval "math" "${MODELS[1]}" "4,5,6,7" 16384 4 "eval_out/full"

# MMLU Pro evaluations
echo "Running MMLU Pro evaluations..."
run_eval "mmlu_pro" "${MODELS[0]}" "4,5,6,7" 4096 4 "eval_out/full"
run_eval "mmlu_pro" "${MODELS[1]}" "4,5,6,7" 4096 4 "eval_out/full"

echo "All evaluations completed!"
echo "Results are saved in result/ directory"
echo "Logs are saved in eval_out/full/ directory"