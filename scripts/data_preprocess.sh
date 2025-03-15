#!/bin/bash

LOCAL_DATA_DIR="/Users/little1d/Desktop/Code/trl4llm/data/ruozhiba"
CACHE_DIR="/Users/little1d/Desktop/Code/trl4llm/data/cache"


python "./data_preprocess/ruozhiba.py" \
    --local_dir "$LOCAL_DATA_DIR" \
    --cache_dir "$CACHE_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 数据预处理完成，结果保存在: $LOCAL_DATA_DIR"
else
    echo "❌ 数据预处理失败"
    exit 1
fi