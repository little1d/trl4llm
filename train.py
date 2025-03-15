from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)  # patch GRPO Algo
from unsloth import is_bfloat16_supported
import re
from sentence_transformers import SentenceTransformer, util

from trl import GRPOConfig, GRPOTrainer
from trl4llm import load_custom_dataset
from swanlab.integration.transformers import SwanLabCallback

swanlab_callback = SwanLabCallback(
    project="trl4llm",
    experiment_name="GRPO-qwen2.5-3b-ruozhi",
    description="弱智吧数据",
)

max_seq_length = 1024
lora_rank = 64
MODEL_PATH = "/mnt/hwfile/ai4chem/share/rl4llms/qwen2.5-3b"

# 从 HuggingFace 加载 Qwen2.5-3B-Instruct 模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # LoRA秩
    target_modules=[  # 应用LoRA的目标模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,  # LoRA缩放系数
    use_gradient_checkpointing="unsloth",  # 启用梯度检查点以支持长序列
    random_state=666,  # 随机种子
)

# 加载数据集
dataset = load_custom_dataset("ruozhiba_qa.json")

# 加载 Sentence Transformers 模型
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


# 语义相似度奖励
def semantic_similarity_reward_func(
    prompts, completions, answer, **kwargs
) -> list[float]:
    responses = [
        completion[0]['content'].strip() for completion in completions
    ]
    answer = [a.strip() for a in answer]

    # 计算相似度
    similarities = util.cos_sim(
        semantic_model.encode(responses), semantic_model.encode(answer)
    )

    rewards = []
    for (
        sim
    ) in (
        similarities.diagonal().tolist()
    ):  # 取对角线上的值（单个样本的相似度）
        if sim > 0.9:
            rewards.append(2.0)  # 非常接近
        elif sim > 0.7:
            rewards.append(1.5)  # 相关性较高
        elif sim > 0.5:
            rewards.append(1.0)  # 可能部分正确
        else:
            rewards.append(0.0)  # 相关性低

    return rewards


# 严格格式奖励：必须完全匹配 <reasoning>...</reasoning><answer>...</answer>
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]


# 软格式奖励：只需包含 <reasoning> 和 <answer> 部分
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.search(pattern, r) else 0.0 for r in responses]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


training_args = GRPOConfig(
    use_vllm=True,  # 使用vLLM加速推理
    learning_rate=5e-6,  # 学习率
    adam_beta1=0.9,  # Adam优化器参数
    adam_beta2=0.99,
    weight_decay=0.1,  # 权重衰减
    warmup_ratio=0.1,  # 学习率预热比例
    lr_scheduler_type="cosine",  # 学习率调度策略
    optim="adamw_8bit",  # 8位Adam优化器
    logging_steps=1,
    bf16=is_bfloat16_supported(),  # 根据硬件支持选择精度
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,  # batch size,你计算资源够的话，可以设置高一点
    gradient_accumulation_steps=1,  # 累计1步后更新一次参数
    num_generations=8,  # 每次生成的候选数
    max_prompt_length=256,  # 输入最大长度
    max_completion_length=200,  # 生成最大长度
    max_steps=200,  # 最大训练步数
    save_steps=50,  # 保存间隔
    max_grad_norm=0.1,  # 梯度裁剪阈值
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[  # 奖励函数列表
        xmlcount_reward_func,  # XML结构奖励
        soft_format_reward_func,  # 宽松格式奖励
        strict_format_reward_func,  # 严格格式奖励
        semantic_similarity_reward_func,  # 语义相似奖励
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[swanlab_callback],
)
trainer.train()  # 启动训练

model.save_lora("grpo_saved_lora")
