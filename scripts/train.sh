#!/bin/bash

# config name in the config_map
CONFIG_NAME="ruozhiba"

# unsloth don't support multi-gpu 😭
export USE_PYTORCH_KERNEL_CACHE=0
export CUDA_VISIBLE_DEVICES="0"

# Activate conda environment if needed
source /fs-computility/llmit_d/shared/baitianyi/miniconda3/etc/profile.d/conda.sh
conda activate trl

# Run training
python -m trl4llm.trainer.grpo \
    --config "$CONFIG_NAME" \