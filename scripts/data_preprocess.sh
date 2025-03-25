#!/bin/bash

# 修改数据集和缓存安装地址，注意 Local data dir 要加上一段自定义的目录
# 如安装到 Home 目录 data 文件夹下 
# LOCAL_DATA_DIR="~/data/ruozhiba"
LOCAL_DATA_DIR="/fs-computility/llmit_d/shared/baitianyi/datasets/stack_exchange"

# 如果删去这个参数，默认为 ~/.cache
CACHE_DIR="/fs-computility/llmit_d/shared/baitianyi/cache"

# 修改脚本文件
python "../data_preprocess/stack_exchange.py" \
    --local_dir "$LOCAL_DATA_DIR" \
    --cache_dir "$CACHE_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 数据预处理完成，结果保存在: $LOCAL_DATA_DIR"
else
    echo "❌ 数据预处理失败"
    exit 1
fi