"""
Generic PPO Trainer implementation
"""

import argparse
from trl import PPOTrainer
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from trl4llm.configs import Gsm8kPPOConfig
import shutil

def check_disk_space(min_space_gb=5):
    total, used, free = shutil.disk_usage("/")
    if free < min_space_gb * (1 << 30):
        raise RuntimeError(f"Insufficient disk space. Need at least {min_space_gb}GB free")


def get_config_class(config_name: str):
    """Get configuration class by name"""
    config_map = {"gsm8k": Gsm8kPPOConfig}
    if config_name not in config_map:
        raise ValueError(f"Unknown config name: {config_name}")

    return config_map[config_name]


def train(config_name):
    """
    Main training function for PPO

    Args:
        config_name: Name of the configuration to use
    """
    # Load configuration
    config_class = get_config_class(config_name)
    config = config_class()

    # Initialize components
    policy_model, ref_model, value_model, reward_model, tokenizer = (
        config.initialize_models()
    )

    # Load datasets from parquet
    train_dataset = load_dataset(
        "parquet",
        data_files={"train": config.dataset_config["train_path"]},
        split="train",
    )
    test_dataset = load_dataset(
        "parquet",
        data_files={"test": config.dataset_config["test_path"]},
        split="test",
    )

    # pre-tokenize
    def prepare_datasets(dataset, tokenizer):
        # dataset_text_field = "prompt"
        check_disk_space()
        def tokenize(element):
            prompt = element["prompt"]
            
            # 处理嵌套列表格式的对话
            if isinstance(prompt, list):
                # 展平嵌套列表结构
                flat_prompt = []
                for msg in prompt:
                    if isinstance(msg, list):
                        flat_prompt.extend(msg)
                    else:
                        flat_prompt.append(msg)
                
                # 转换为文本
                text = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    if isinstance(msg, dict) 
                    else str(msg) 
                    for msg in flat_prompt
                ])
            else:
                # 处理普通文本格式
                text = str(prompt) if prompt is not None else ""
            outputs = tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=1000, 
                return_tensors=None,
            )            
            return {"input_ids": outputs["input_ids"]}
        
        tokenized_dataset = dataset.map(tokenize, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(
            [col for col in tokenized_dataset.column_names if col != "input_ids"]
        )
        return tokenized_dataset

    train_dataset = prepare_datasets(train_dataset, tokenizer)
    test_dataset = prepare_datasets(test_dataset, tokenizer)
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        args=config.training_args,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[
            SwanLabCallback(
                project="trl4llm",
                experiment_name=f"PPO-{config_name}",
                description=f"{config_name} dataset training",
            )
        ],
    )

    # Start training
    trainer.train()

    # Save model
    policy_model.save_pretrained(config.save_dir)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run PPO training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the configuration to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_name=args.config)
