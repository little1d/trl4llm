"""
Generic GRPO Trainer implementation
"""

import argparse
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback

from trl4llm.configs import RuozhibaConfig, Gsm8kConfig


def get_config_class(config_name: str):
    """Get configuration class by name"""
    config_map = {"ruozhiba": RuozhibaConfig, "gsm8k", Gsm8kConfig}
    if config_name not in config_map:
        raise ValueError(f"Unknown config name: {config_name}")

    return config_map[config_name]


def train(config_name):
    """
    Main training function

    Args:
        config_name: Name of the configuration to use
    """
    # Load configuration
    config_class = get_config_class(config_name)
    config = config_class()

    # Initialize components
    model, tokenizer = config.initialize_model()
    reward_functions = config.initialize_reward_functions()

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

    # Initialize trainer based on method
    # Reference:
    #   https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        reward_funcs=reward_functions,
        callbacks=[
            SwanLabCallback(
                project="trl4llm",
                experiment_name=f"GRPO-{config_name}",
                description=f"{config_name} dataset training",
            )
        ],
    )
    trainer.train()
    # save lora adapter
    model.save_pretrained(config.save_dir)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the configuration to use, remember registry it first!",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_name=args.config)
