"""
Configuration class for ruozhiba dataset
"""
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)
import torch
from sentence_transformers import SentenceTransformer
from trl import GRPOConfig
from trl4llm.reward_score.ruozhiba import (
    strict_format_reward_func,
    soft_format_reward_func,
    semantic_similarity_reward_func,
    xmlcount_reward_func,
)


class RuozhibaGRPOConfig:
    def __init__(self):
        # Training parameters
        self.training_args = GRPOConfig(
            # use_vllm=True,
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            # 日志存储
            logging_dir="./gpro_logs",
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            num_generations=8,
            max_prompt_length=256,
            max_completion_length=200,
            max_steps=200,
            save_steps=50,
            max_grad_norm=0.1,
            report_to="none",
        )

        # Dataset configuration
        self.dataset_config = {
            "train_path": "/fs-computility/llmit_d/shared/baitianyi/datasets/ruozhiba/train.parquet",
            "test_path": "/fs-computility/llmit_d/shared/baitianyi/datasets/ruozhiba/test.parquet",
        }

        # Save path
        self.save_dir = "./rouzhiba_lora"

    def initialize_model(self):
        """Initialize model with LoRA configuration"""
        lora_rank = 64
        # model local path
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-3B-Instruct",
            max_seq_length=1024,
            load_in_4bit=True,  # 4 位量化
            # fast_inference=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.7,
        )
        # lora adapter
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=666,
        )
        return model, tokenizer

    def initialize_reward_functions(self):
        """Initialize reward functions with semantic model"""
        semantic_model = SentenceTransformer(
            # semantic_model local path
            "/fs-computility/llmit_d/shared/baitianyi/model/all-MiniLM-L6-v2"
        )

        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            # 适配 GRPOTrainer 参数结构，隐藏 semantic_model 实现细节
            lambda prompts, completions, answer: semantic_similarity_reward_func(
                prompts, completions, semantic_model, answer
            ),
        ]
