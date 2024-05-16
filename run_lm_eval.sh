#!/bin/bash

# Run your command here
# lm_eval --model vllm \
#     --model_args pretrained=mistralai/Mixtral-8x7B-v0.1,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8 \
#     --tasks "mmlu" \
#     --batch_size auto

lm_eval --model vllm --model_args pretrained=mistralai/Mixtral-8x7B-Instruct-v0.1,tensor_parallel_size=8 --tasks mmlu --num_fewshot 5

# Print "done" at the end
echo "done"
