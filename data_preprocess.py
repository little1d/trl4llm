from datasets import Dataset
import json

SYSTEM_PROMPT = """
请使用中文按以下格式回答问题:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


# 读取自定义数据集 ruozhiba.json
def load_custom_dataset(file_path="ruozhiba.json") -> Dataset:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理数据为符合训练格式
    processed_data = []
    for item in data:
        instruction = item["instruction"]
        output = item["output"]

        # 设定 prompt 格式（符合 chat 训练格式）
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        processed_data.append({"prompt": prompt, "answer": output})

    # 转换为 Hugging Face Dataset
    dataset = Dataset.from_list(processed_data)
    return dataset
