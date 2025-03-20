from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import PPOConfig

max_prompt_length = 256
max_seq_length = 1024
lora_rank = 32


class Gsm8kPPOConfig:
    def __init__(self):
        # Training parameters
        self.training_args = PPOConfig(
            learning_rate=5e-6,
            batch_size=1,
            mini_batch_size=1,
            ppo_epochs=1,
            gradient_accumulation_steps=1,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            log_with="wandb",
            logging_dir="./ppo_logs",
            max_grad_norm=0.1,
            output_dir="outputs",
        )

        # Dataset configuration
        self.dataset_config = {
            "train_path": "/fs-computility/llmit_d/shared/baitianyi/datasets/gsm8k/train.parquet",
            "test_path": "/fs-computility/llmit_d/shared/baitianyi/datasets/gsm8k/test.parquet",
        }

        # Model paths
        self.model_config = {
            "policy_model_path": "/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-3B-Instruct",
            "reward_model_path": "/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-3B-Instruct",
        }

        # Save path
        self.save_dir = "./gsm8k_ppo_lora"

    def initialize_models(self):
        """Initialize policy, reward, value and reference models"""
        # Initialize policy model with LoRA
        policy_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config["policy_model_path"],
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.6,
        )

        policy_model = FastLanguageModel.get_peft_model(
            policy_model,
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
            random_state=3407,
        )

        # Initialize value and reward models (using same base model)
        value_model = FastLanguageModel.from_pretrained(
            model_name=self.model_config["reward_model_path"],
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )[0]

        reward_model = FastLanguageModel.from_pretrained(
            model_name=self.model_config["reward_model_path"],
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )[0]

        # Reference model is None when using PEFT
        ref_model = None

        return policy_model, ref_model, value_model, reward_model, tokenizer
