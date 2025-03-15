"""
Configuration class for ruozhiba dataset
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from sentence_transformers import SentenceTransformer
from reward_score.ruozhiba import (
    strict_format_reward_func,
    soft_format_reward_func,
    semantic_similarity_reward_func,
    xmlcount_reward_func,
)
from trl import GRPOConfig

max_seq_length = 1024  # 模型支持的最大序列长度
lora_rank = 64  # LoRA的秩，值越大模型能力越强但速度越慢


class RuozhibaConfig:
    def __init__(self):
        # Training parameters
        self.training_args = GRPOConfig(
            use_vllm=True,
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            logging_dir='./gpro_logs',
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=8,
            max_prompt_length=256,
            max_completion_length=200,
            max_steps=200,
            save_steps=50,
            max_grad_norm=0.1,
            report_to="none",
            output_dir="grpo_saved_lora",
        )

        # Dataset configuration
        self.dataset_config = {
            "train_path": "./data/processed/train.parquet",
            "test_path": "./data/processed/test.parquet",
        }

        # Save path
        self.save_dir = "./ruozhiba_grpo"

    def initialize_model(self):
        """Initialize model with LoRA configuration"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="LooksJuicy/ruozhiba",
            max_seq_length=1024,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.6,
        )
        # 为模型添加LoRA适配器
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
        return model, tokenizer

    def initialize_reward_functions(self):
        """Initialize reward functions with semantic model"""
        semantic_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            lambda completions, ground_truths: semantic_similarity_reward_func(
                completions, semantic_model, ground_truths
            ),
        ]
