from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)
import torch
from trl import GRPOConfig
from trl4llm.reward_score.gsm8k import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)

max_prompt_length = 256
max_seq_length = 1024
lora_rank = 32


class Gsm8kConfig:
    def __init__(self):
        # Training parameters
        self.training_args = GRPOConfig(
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=1,
            logging_dir="./gpro_logs",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,  # Increase to 4 for smoother training
            num_generations=6,  # Decrease if out of memory
            max_prompt_length=max_prompt_length,
            max_completion_length=max_seq_length - max_prompt_length,
            # num_train_epochs = 1, # Set to 1 for a full training run
            max_steps=250,
            save_steps=250,
            max_grad_norm=0.1,
            report_to="wandb",  # Can use Weights & Biases
            output_dir="outputs",
        )

        # Dataset configuration
        self.dataset_config = {
            "train_path": "/fs-computility/llmit_d/shared/baitianyi/datasets/gsm8k/train.parquet",
            "test_path": "/fs-computility/llmit_d/shared/baitianyi/datasets/gsm8k/test.parquet",
        }

        # Save path
        self.save_dir = "./gsm8k_grpo_lora"

        def initialize_model(self):
            """Initialize model with LoRA configuration"""
            # model locan path
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-3B-Instruct",
                max_seq_length=max_seq_length,
                load_in_4bit=True,  # False for LoRA 16bit
                # fast_inference = True, # Enable vLLM fast inference
                max_lora_rank=lora_rank,
                gpu_memory_utilization=0.6,  # Reduce if out of memory
            )
            # lora adapter
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # Remove QKVO if out of memory
                lora_alpha=lora_rank,
                use_gradient_checkpointing="unsloth",  # Enable long context finetuning
                random_state=3407,
            )

            return model, tokenizer

        def initialize_reward_functions(self):
            """Initialize reward functions"""
            return [
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
            ]
