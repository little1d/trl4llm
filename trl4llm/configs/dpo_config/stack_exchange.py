from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import DPOConfig

max_prompt_length = 512
max_seq_length = 1024
lora_rank = 8


class StackExchangeDPOConfig:
    def __init__(self):
        # Training parameters
        self.training_args = DPOConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            num_train_epochs=2,
            learning_rate=5e-6,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
            report_to="wandb",
            max_length=1024,
            max_prompt_length=512,
            beta=0.1,
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
            use_rslora=False,
            use_gradient_checkpointing="unsloth",
        )

        return model, tokenizer
