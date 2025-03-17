import os
import datasets
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--cache_dir", default=None)

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, 'main',cache_dir=args.cache_dir)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    SYSTEM_PROMPT = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. Put the reasoning process in the <think></think> tag, and put the answer in the <answer></answer> tag."
        r"""
    Then answer the question in English in the following format:
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
            question = example.pop("question")
            answer_raw = example.pop("answer")
            prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
            data = {
                "prompt": prompt,
                "answer": answer_raw
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
