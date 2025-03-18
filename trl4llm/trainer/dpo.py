"""
Generic DPO Trainer implementation
"""

import argparse
import os
from trl import DPOTrainer
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback

from trl4llm.configs import StackExchangeDPOConfig


def get_config_class(config_name: str):
    """Get configuration class by name"""
    config_map = {"stack_exchange": StackExchangeDPOConfig}
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

    # Initialize DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # use base model as reference
        args=config.training_args,
        beta=config.training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        max_prompt_length=config.training_args.max_prompt_length,
        max_length=config.training_args.max_ength,
    )

    # Add swanlab callback
    dpo_trainer.add_callback(
        SwanLabCallback(
            project="trl4llm",
            experiment_name=f"DPO-{config_name}",
            description=f"{config_name} dataset training",
        )
    )

    # Start training
    dpo_trainer.train()

    # Save model
    dpo_trainer.save_model(config.save_dir)

    # Save pre-trained model
    final_checkpoint_dir = os.path.join(config.save_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(final_checkpoint_dir)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the configuration to use, remember registry it first!",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_name=args.config)
