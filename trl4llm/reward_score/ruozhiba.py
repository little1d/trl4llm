"""
Reward score calculate functions for ruozhiba dataset
"""

import re
from typing import List
from sentence_transformers import util


def strict_format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    严格格式检查奖励函数
    检查是否包含 <think> 和 <answer> 标签
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [1.0 if re.search(pattern, c) else 0.0 for c in completions]


def soft_format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    宽松格式检查奖励函数
    检查是否包含 <think> 或 <answer> 标签
    """
    pattern = r"(<think>.*?</think>|<answer>.*?</answer>)"
    return [0.5 if re.search(pattern, c) else 0.0 for c in completions]


def semantic_similarity_reward_func(
    completions: List[str],
    semantic_model,  # 语义模型从外部传入
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    """
    语义相似度奖励函数
    使用外部传入的语义模型计算相似度
    """
    # 计算嵌入
    completion_embeddings = semantic_model.encode(completions)
    gt_embeddings = semantic_model.encode(ground_truths)

    # 计算相似度
    similarities = util.cos_sim(completion_embeddings, gt_embeddings)
    return [float(s) for s in similarities.diagonal()]


def xmlcount_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    XML 标签计数奖励函数
    检查 <think> 和 <answer> 标签的数量
    """
    rewards = []
    for c in completions:
        think_count = len(re.findall(r"<think>", c))
        answer_count = len(re.findall(r"<answer>", c))
        rewards.append(min(think_count + answer_count, 1.0))
    return rewards
