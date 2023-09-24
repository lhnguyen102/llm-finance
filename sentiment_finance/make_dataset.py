import json
import warnings

import datasets
from datasets import load_dataset
from tqdm import tqdm
from tqdm.notebook import tqdm

from tokenizer import Tokenizer

warnings.filterwarnings("ignore")


def make_label(x):
    if x < -0.1:
        return "negative"
    elif x >= -0.1 and x < 0.1:
        return "neutral"
    elif x >= 0.1:
        return "positive"


def add_instructions(x):
    if x == "post":
        return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    else:
        return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."


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
            example = json.loads(line)
            feature = preprocess(tokenizer_model, example)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def main():
    # FPB
    dic = {0: "negative", 1: "neutral", 2: "positive"}
    fpb_datasets = load_dataset("financial_phrasebank", "sentences_50agree")
    fpb_datasets = fpb_datasets["train"]
    fpb_datasets = fpb_datasets.to_pandas()
    fpb_datasets.columns = ["input", "output"]
    fpb_datasets["output"] = fpb_datasets["output"].apply(lambda x: dic[x])
    fpb_datasets[
        "instruction"
    ] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    fpb_datasets = datasets.Dataset.from_pandas(fpb_datasets)
    fpb_datasets = fpb_datasets.train_test_split(seed=42)["train"]
    train_dataset = datasets.concatenate_datasets([fpb_datasets] * 6)

    # FiQA SA
    dataset = load_dataset("pauri32/fiqa-2018")
    dataset = datasets.concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )
    dataset = dataset.to_pandas()
    dataset["output"] = dataset.sentiment_score.apply(make_label)
    dataset["instruction"] = dataset.format.apply(add_instructions)
    dataset = dataset[["sentence", "output", "instruction"]]
    dataset.columns = ["input", "output", "instruction"]
    dataset = datasets.Dataset.from_pandas(dataset)
    dataset = dataset.train_test_split(0.226, seed=42)["train"]

    tmp_dataset = datasets.concatenate_datasets([dataset] * 21)
    train_dataset = datasets.concatenate_datasets([train_dataset, tmp_dataset])

    # TFNS
    social_media_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    social_media_dataset = social_media_dataset["train"]
    social_media_dataset = social_media_dataset.to_pandas()
    social_media_dataset["label"] = social_media_dataset["label"].apply(lambda x: dic[x])
    social_media_dataset[
        "instruction"
    ] = "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    social_media_dataset.columns = ["input", "output", "instruction"]
    social_media_dataset = datasets.Dataset.from_pandas(social_media_dataset)

    tmp_dataset = datasets.concatenate_datasets([social_media_dataset] * 2)
    train_dataset = datasets.concatenate_datasets([train_dataset, tmp_dataset])

    # NWGI
    finance_dataset = load_dataset("oliverwang15/news_with_gpt_instructions")
    finance_dataset = finance_dataset["train"].to_pandas()
    finance_dataset["output"] = finance_dataset["label"]
    finance_dataset["input"] = finance_dataset["news"]
    finance_dataset[
        "instruction"
    ] = "What is the sentiment of this news? Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/mildly positive/moderately positive/strong positive}, then provide some short reasons."
    finance_dataset = finance_dataset[["input", "output", "instruction"]]
    finance_dataset = datasets.Dataset.from_pandas(finance_dataset)

    train_dataset = datasets.concatenate_datasets([train_dataset, finance_dataset])

    all_dataset = train_dataset.shuffle(seed=42)

    #
    data_list = []
    for item in all_dataset.to_pandas().itertuples():
        tmp = {}
        tmp["instruction"] = item.instruction
        tmp["input"] = item.input
        tmp["output"] = item.output
        data_list.append(tmp)

    with open("sentiment_finance/data/dataset_new.jsonl", "w") as f:
        for example in tqdm(data_list, desc="formatting.."):
            f.write(json.dumps(format_example(example)) + "\n")

    # Tokenize
    model_name = "tokenizer.model"
    jsonl_path = "sentiment_finance/data/dataset_new.jsonl"
    save_path = "sentiment_finance/sen_dataset"
    max_seq_length = 512
    skip_overlength = True
    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(jsonl_path, max_seq_length, model_name, skip_overlength)
    )
    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()
