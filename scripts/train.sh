#!/bin/bash

export 

# Activate conda environment if needed
source /fs-computility/llmit_d/shared/baitianyi/miniconda3/etc/profile.d/conda.sh
conda activate trl

# Run training
python trl4llm/train.py
