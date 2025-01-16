#!/bin/bash

# Define the range for gamma values
for run in {1..3}; do
  echo "Running HF BASE, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "meta-llama/Llama-3.1-8B" \
    --dataset humaneval \
    --backend hf \
    --greedy \
    --force-base-prompt \
    --device_map "auto" \
    --trust_remote_code true

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done

for gamma in {2..10}; do
  # Run the command 3 times for each gamma value
  for run in {1..3}; do
    echo "Running command BASE with gamma=$gamma, run=$run"
    start_time=$(date +%s)
    
    python evalplus/evaluate.py \
      --model "meta-llama/Llama-3.1-8B" \
      --dataset humaneval \
      --backend spec \
      --greedy \
      --force-base-prompt \
      --device_map "auto" \
      --trust_remote_code true \
      --gamma $gamma

    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
  done
done
