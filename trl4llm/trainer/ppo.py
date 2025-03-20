"""
Generic PPO Trainer implementation
"""

import argparse
from trl import PPOTrainer
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback
from trl4llm.configs import Gsm8kPPOConfig


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

    # Initialize PPO trainer
    trainer = PPOTrainer(
        args=config.training_args,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
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
