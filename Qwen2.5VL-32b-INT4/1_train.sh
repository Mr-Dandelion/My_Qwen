#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="training_output%100"                  # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         # Core Arguments
         --model_name_or_path $MODEL_PATH \  # [ModelArguments] Model identifier
         --tune_mm_llm True \                # [TrainingArguments] Train LLM or not
         --tune_mm_vision False \            # [TrainingArguments] Train VIT or not
         --tune_mm_mlp False \               # [TrainingArguments] Train MLP or not
         --dataset_use $DATASETS \           # [DataArguments] Dataset specification
         --output_dir $OUTPUT_DIR \          # Output directory for checkpoints
         --cache_dir $CACHE_DIR \            # [TrainingArguments] Model cache location
         
         # Precision & Memory
         --bf16 \                            # Use bfloat16 precision (Ampere+ GPUs)
         --per_device_train_batch_size 4 \   # Batch size per GPU
         --gradient_accumulation_steps 4 \   # Effective batch size multiplier
         
         # Learning Rate Configuration
         --learning_rate 2e-7 \              # Base learning rate
         --mm_projector_lr 1e-5 \            # [TrainingArguments] Projector-specific LR
         --vision_tower_lr 1e-6 \            # [TrainingArguments] Vision encoder LR
         --optim adamw_torch \               # [TrainingArguments] Optimizer selection
         
         # Sequence Configuration
         --model_max_length 4096 \           # [TrainingArguments] Max sequence length
         --data_flatten True \               # [DataArguments] Concatenate batch sequences
         --data_packing True \               # [DataArguments] Using packing data
         
         # Image Processing
         --max_pixels 576\*28\*28 \               # [DataArguments] Max image pixels (H*W) for image
         --min_pixels 16\*28\*28 \                # [DataArguments] Min image pixels for image
         # Video Processing
         --base_interval 2 \                      # [DataArguments] Sampling time interval (seconds) between frames
         --video_max_frames 8 \                   # [DataArguments] Max frames per video
         --video_min_frames 4 \                   # [DataArguments] Min frames per video
         --video_max_frame_pixels 1664\*28\*28 \  # [DataArguments] Max pixels within a frame
         --video_min_frame_pixels 256\*28\*28 \   # [DataArguments] Min pixels within a frame
         
         # Training Schedule
         --num_train_epochs 3 \              # Total training epochs
         --warmup_ratio 0.03 \               # LR warmup proportion
         --lr_scheduler_type "cosine" \      # Learning rate schedule
         --weight_decay 0.01 \               # L2 regularization strength
         
         # Logging & Checkpoints
         --logging_steps 10 \               # Log metrics interval
         --save_steps 500 \                 # Checkpoint save interval
         --save_total_limit 3 \             # Max checkpoints to keep
         
         # Advanced Options
         --deepspeed zero3.json \           # DeepSpeed configuration