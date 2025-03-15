#!/bin/bash

CONFIG_NAME="ruozhiba"

# Activate conda environment if needed
source /fs-computility/llmit_d/shared/baitianyi/miniconda3/etc/profile.d/conda.sh
conda activate trl

# Run training
python -m trl4llm.trainer.grpo \
    --config "$CONFIG_NAME" \