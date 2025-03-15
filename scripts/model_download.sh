#!/bin/bash

MODEL_REPO="sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH="/fs-computility/llmit_d/shared/baitianyi/model/all-MiniLM-L6-v2"
TOKEN="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"

python model_download.py \
    --model_repo "$MODEL_REPO" \
    --save_path "$SAVE_PATH" \
    $( [ -n "$TOKEN" ] && echo "--token $TOKEN" )