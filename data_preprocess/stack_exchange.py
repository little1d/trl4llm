import os
import datasets
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/stack_exchange")
    parser.add_argument("--cache_dir", default=None)

    args = parser.parse_args()

    data_source = "lvwerra/stack-exchange-paired"

    dataset = datasets.load_dataset(data_source, cache_dir=args.cache_dir)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def make_map_fn(split):

        def process_fun(example, idx):
            question = example.pop("question")
            instruction_following = "\n\nAnswer: "
            chosen = example.pop("response_j")  # rated better than k
            rejected = example.pop("response_k")  # rated worse than j
            prompt = {
                "role": "user",
                "content": question + "" + instruction_following,
            }
            data = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
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
