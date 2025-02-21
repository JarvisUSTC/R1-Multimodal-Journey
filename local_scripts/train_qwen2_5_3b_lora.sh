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

EXP_NAME="Qwen2_5-VL-3B-LORA-GRPO-20k"

export WANDB_PROJECT="R1-multimodal-Jiawei"
export WANDB_NAME=$EXP_NAME
export WANDB_MODE="online"  # 可选：如果不想上传日志
export WANDB_RUN_GROUP="experiment_001"  # 可选：让多次实验归类
export WANDB_DISABLE_SERVICE=true  # 解决多进程 `wandb` 可能导致的冲突

# 创建results/Qwen2_5-VL-3B-LORA-GRPO-20k/
OUTPUT_PATH=results/$EXP_NAME
mkdir -p $OUTPUT_PATH

export LOG_PATH=$OUTPUT_PATH/train_record.log

torchrun \
    --nproc_per_node="3" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero2_v100.json \
    --output_dir $OUTPUT_PATH \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name /home/t-jiaweiwang/Project/MLLM-PostTraining/R1-Multimodal-Journey/geo_data \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --use_peft True \
    --lora_target_modules down_proj o_proj k_proj q_proj gate_proj up_proj v_proj \
    --logging_steps 1 \
    --fp16 \
    --torch_dtype float16 \
    --gradient_checkpointing true \
    --attn_implementation eager \
    --max_pixels 1000000 \
    --save_total_limit 30 \
    --num_train_epochs 1 \
    --run_name $EXP_NAME \
    >> $OUTPUT_PATH/train_lora.log 2>&1
