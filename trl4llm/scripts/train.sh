#!/bin/bash

# Activate conda environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Run training
python trl4llm/train.py
