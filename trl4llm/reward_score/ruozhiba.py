"""
Reward score calculate functions for ruozhiba dataset
"""

import re
from typing import List
from sentence_transformers import util, SentenceTransformer


def strict_format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    严格格式检查奖励函数
    检查是否包含 <think> 和 <answer> 标签
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

# 软格式奖励：只需包含 <think> 和 <answer> 部分
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.search(pattern, r) else 0.0 for r in responses]

#语义相似度奖励
def semantic_similarity_reward_func(prompts, completions, semantic_model, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'].strip() for completion in completions]
    answer = [a.strip() for a in answer]

    # 计算相似度
    similarities = util.cos_sim(semantic_model.encode(responses), semantic_model.encode(answer))

    rewards = []
    for sim in similarities.diagonal().tolist():  # 取对角线上的值（单个样本的相似度）
        if sim > 0.9:
            rewards.append(2.0)  # 非常接近
        elif sim > 0.7:
            rewards.append(1.5)  # 相关性较高
        elif sim > 0.5:
            rewards.append(1.0)  # 可能部分正确
        else:
            rewards.append(0.0)  # 相关性低

    return rewards


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]