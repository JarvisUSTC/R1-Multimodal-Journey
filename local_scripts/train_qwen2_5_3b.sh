#!/bin/bash

# 添加默认参数
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}

DISTRIBUTED_ARGS="
    --nproc_per_node 3 \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"


torchrun \
    --nproc_per_node="3" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir results \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name /home/t-jiaweiwang/Project/MLLM-PostTraining/R1-Multimodal-Journey/geo_data \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --fp16 \
    --torch_dtype float16 \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --max_pixels 1000000 \
    --save_total_limit 30 \
    --num_train_epochs 1 \
    --run_name Qwen2_5-VL-3B-GRPO-20k \
    >> train.log 2>&1
