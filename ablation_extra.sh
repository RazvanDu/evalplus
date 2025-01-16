#!/bin/bash

#for run in {1..3}; do
#  echo "Running big command with gamma=3, run=$run"
#  start_time=$(date +%s)
  
#  python evalplus/evaluate.py \
#    --model "meta-llama/Llama-3.1-70B-Instruct" \
#    --dataset humaneval \
#    --backend spec \
#    --greedy \
#    --device_map "auto" \
#    --trust_remote_code true \
#    --gamma 3

#  end_time=$(date +%s)
#  execution_time=$((end_time - start_time))
#  echo "Execution time for gamma=3, run=$run: ${execution_time}s"
#done

for run in {1..3}; do
  echo "Running big command HF, run=$run"
  start_time=$(date +%s)
  
  python evalplus/evaluate.py \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --dataset humaneval \
    --backend hf \
    --greedy \
    --device_map "auto" \
    --trust_remote_code true \

  end_time=$(date +%s)
  execution_time=$((end_time - start_time))
  echo "Execution time for HF, run=$run: ${execution_time}s"
done