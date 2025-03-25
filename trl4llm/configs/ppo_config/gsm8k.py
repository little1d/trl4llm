from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trl import PPOConfig

max_prompt_length = 256
max_seq_length = 1024
lora_rank = 32


class Gsm8kPPOConfig:
    def __init__(self):
        # Training parameters
        # Reference
        #   https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_config.py
        self.training_args = PPOConfig(
            learning_rate=5e-6,
            mini_batch_size=1,
            num_ppo_epochs=1,
            gradient_accumulation_steps=1,
            report_to="wandb",
            logging_dir="./ppo_logs",
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
        """Initialize models using traditional PEFT and TRL"""
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["policy_model_path"],
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize policy model with LoRA
        policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_config["policy_model_path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # LoRA config
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )

        policy_model = get_peft_model(policy_model, peft_config)
        policy_model.config.use_cache = False

        # Initialize value and reward models
        value_model = AutoModelForCausalLM.from_pretrained(
            self.model_config["reward_model_path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        reward_model = AutoModelForCausalLM.from_pretrained(
            self.model_config["reward_model_path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Reference model is None when using PEFT
        ref_model = None

        return policy_model, ref_model, value_model, reward_model, tokenizer