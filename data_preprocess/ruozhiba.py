import os
import datasets
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/ruozhiba")
    parser.add_argument("--cache_dir", default=None)

    args = parser.parse_args()

    data_source = "LooksJuicy/ruozhiba"

    dataset = datasets.load_dataset(data_source, cache_dir=args.cache_dir)
    # ruozhiba数据库没做 test 数据集，需要手动切分。
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"首先把推理过程想象成内心独白，然后提供最终的答案。推理过程放在<think></think>标签中，答案放在<answer></answer>标签中。"
        r"""
    然后使用中文按以下格式回答问题:
    <think>
    ...
    </think>
    <answer>
    ...
    </answer>
    """
    )

    def make_map_fn(split):

        def process_fun(example, idx):
            instruction_raw = example.pop("instruction")
            prompt = instruction_raw + "" + instruction_following

            answer_raw = example.pop("output")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "IQ",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": instruction_raw,
                    "answer": answer_raw,
                },
            }

            return data

        return process_fun

    train_dataset = train_dataset.map(
        function=make_map_fn("train"), with_indices=True, num_proc=8
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"), with_indices=True, num_proc=8
    )

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save processed data
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
