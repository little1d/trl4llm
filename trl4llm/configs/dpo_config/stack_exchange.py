from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL

# TODO
PatchFastRL("DPO", FastLanguageModel)

import torch
from trl import DPOConfig

max_prompt_length = 512
max_seq_length = 1024
lora_rank = 8


class StackExchangeDPOConfig:
    def __init__(self):
        # Training parameters
        self.training_args = DPOConfig(
            learning_rate=5e-4,
            beta=0.1,
            weight_decay=0.05,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            optim="paged_adamw_32bit",
            logging_steps=10,
            logging_dir="./dpo_logs",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_use_reentrant=False,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_seq_length - max_prompt_length,
            max_steps=1000,
            save_steps=100,
            eval_steps=100,
            max_grad_norm=1.0,
            report_to="wandb",
            output_dir="outputs",
            seed=0,
            bf16=True,
        )

        # Dataset configuration
        # TODO
        self.dataset_config = {
            "train_path": "path/to/train/data",  # Update with actual path
            "test_path": "path/to/test/data",  # Update with actual path
        }

        # Save path
        self.save_dir = "./stack_exchange_dpo_lora"

    def initialize_model(self):
        """Initialize model with LoRA configuration"""
        # TODO add model name
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.6,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",
        )

        return model, tokenizer
