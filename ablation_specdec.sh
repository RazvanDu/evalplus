#!/bin/bash

for run in {1..2}; do
  echo "Running big command Qwen/Qwen2.5-72B-Instruct with gamma=3, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "Qwen/Qwen2.5-72B-Instruct" \
    --dataset humaneval \
    --backend spec \
    --greedy \
    --device_map "auto" \
    --trust_remote_code true \
    --gamma 3

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for Qwen/Qwen2.5-72B-Instruct gamma=3, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running big command Qwen/Qwen2.5-72B-Instruct HF, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "Qwen/Qwen2.5-72B-Instruct" \
    --dataset humaneval \
    --backend hf \
    --greedy \
    --device_map "auto" \
    --trust_remote_code true \

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for Qwen/Qwen2.5-72B-Instruct HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running HF BASE meta-llama/Llama-3.1-70B, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "meta-llama/Llama-3.1-70B" \
    --dataset humaneval \
    --backend hf \
    --greedy \
    --force-base-prompt \
    --device_map "auto" \
    --trust_remote_code true

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for meta-llama/Llama-3.1-70B HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running command BASE meta-llama/Llama-3.1-70B with gamma=$gamma, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "meta-llama/Llama-3.1-70B" \
    --dataset humaneval \
    --backend spec \
    --greedy \
    --force-base-prompt \
    --device_map "auto" \
    --trust_remote_code true \
    --gamma $gamma

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for meta-llama/Llama-3.1-70B gamma=$gamma, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running HF BASE Qwen/Qwen2.5-72B, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "Qwen/Qwen2.5-72B" \
    --dataset humaneval \
    --backend hf \
    --greedy \
    --force-base-prompt \
    --device_map "auto" \
    --trust_remote_code true

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for Qwen/Qwen2.5-72B HF, run=$run: ${execution_time}s"
done

for run in {1..2}; do
  echo "Running command BASE Qwen/Qwen2.5-72B with gamma=$gamma, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "Qwen/Qwen2.5-72B" \
    --dataset humaneval \
    --backend spec \
    --greedy \
    --force-base-prompt \
    --device_map "auto" \
    --trust_remote_code true \
    --gamma $gamma

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for Qwen/Qwen2.5-72B gamma=$gamma, run=$run: ${execution_time}s"
done