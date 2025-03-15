"""
Generic GRPO Trainer implementation
"""

from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
from swanlab.integration.transformers import SwanLabCallback

from configs import RuozhibaConfig


def get_config_class(config_name: str):
    """Get configuration class by name"""
    config_map = {"ruozhiba": RuozhibaConfig}
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

    # Load datasets
    train_dataset = load_from_disk(config.dataset_config["train_path"])
    test_dataset = load_from_disk(config.dataset_config["test_path"])

    # Initialize trainer based on method
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=GRPOConfig(config.training_args),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        reward_functions=reward_functions,
        callbacks=[
            SwanLabCallback(
                project="trl4llm",
                experiment_name=f"GRPO-{config_name}",
                description=f"{config_name} dataset training",
            )
        ],
    )
    trainer.train()

    model.save_lora(config.save_dir)


if __name__ == "__main__":
    train()
