import json
import logging

import datasets
from tqdm import tqdm

from tokenizer import Tokenizer


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def load_data(data_path: str) -> list:
    try:
        with open(data_path, encoding="utf-8") as f:
            examples = json.load(f)
            return examples
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from {data_path}")
        return []


def save_data(save_path: str, examples: list) -> None:
    with open(save_path, "w") as f:
        for example in tqdm(examples, desc="Formatting.."):
            formatted_example = format_example(example)
            f.write(json.dumps(formatted_example) + "\n")


def preprocess(tokenizer_model, example):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer_model.encode(prompt, bos=False, eos=False)
    target_ids = tokenizer_model.encode(target, bos=False, eos=False)
    target_ids = target_ids + [tokenizer_model.eos_id]
    return {"input_ids": prompt_ids, "labels": target_ids}


def read_jsonl(path, max_seq_length, model_name, skip_overlength=False):
    tokenizer_model = Tokenizer(model_name)
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            text = json.loads(line)
            feature = preprocess(tokenizer_model, text)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def convert_to_ids():
    model_name = "tokenizer.model"
    jsonl_path = "alpaca_finance/alpaca_ft.jsonl"
    save_path = "alpaca_finance/alpaca_dataset"
    max_seq_length = 512
    skip_overlength = True
    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(jsonl_path, max_seq_length, model_name, skip_overlength)
    )
    dataset.save_to_disk(save_path)


def convert_to_jsonl():
    data_path = "./data/alpaca/Cleaned_date.json"
    save_path = "./data/alpaca/alpaca_ft.jsonl"

    examples = load_data(data_path)
    if examples:
        save_data(save_path, examples)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # convert_to_jsonl()
    convert_to_ids()
