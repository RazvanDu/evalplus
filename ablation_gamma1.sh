#!/bin/bash

for gamma in {1..1}; do
  # Run the command 3 times for each gamma value
  for run in {1..3}; do
    echo "Running command with gamma=$gamma, run=$run"
    start_time=$(date +%s)
    
    python evalplus/evaluate.py \
      --model "meta-llama/Llama-3.1-8B-Instruct" \
      --dataset humaneval \
      --backend spec \
      --greedy \
      --device_map "auto" \
      --trust_remote_code true \
      --gamma $gamma

    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Execution time for gamma=$gamma, run=$run: ${execution_time}s"
  done
done

for gamma in {1..1}; do
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
