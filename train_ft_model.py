import os

import datasets
import fire
import torch

from config import FinetuningModelConfig, NetworkConfig
from finetune_dataset import batch_iterator
from finetune_model import FinetuningModel
from sentiment_finance.test import test_fpb
from alpaca_finance.test import test_qna
from tokenizer import Tokenizer


def train(dataset_name: str):
    """Finetuning API"""
    # Cuda setup
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    print(f"Fine tuning model with using dataset {dataset_name}")
    if dataset_name == "news":
        dataset = datasets.load_from_disk("./sentiment_finance/sen_dataset")
    elif dataset_name == "alpaca":
        dataset = datasets.load_from_disk("./alpaca_finance/alpaca_dataset")
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    # Input
    cfg = FinetuningModelConfig(dataset_name=dataset_name)
    pt_cfg = NetworkConfig()

    # Load dataset
    dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)
    train_dataloader = batch_iterator(
        dataset=dataset["train"],
        batch_size=cfg.batch_size,
        device=cfg.device,
        pretrained_max_len=pt_cfg.max_seq_len,
        pad_token_id=pt_cfg.padding_idx,
        loss_ignore_index=pt_cfg.loss_ignore_index,
        num_workers=0,
    )
    val_dataloader = batch_iterator(
        dataset=dataset["test"],
        batch_size=cfg.batch_size,
        device=cfg.device,
        pretrained_max_len=pt_cfg.max_seq_len,
        pad_token_id=pt_cfg.padding_idx,
        loss_ignore_index=pt_cfg.loss_ignore_index,
        num_workers=0,
    )
    dataloader = {"train": train_dataloader, "val": val_dataloader}

    # Fine-tuning model
    tokenizer_model = Tokenizer("tokenizer.model")
    ft_model = FinetuningModel(cfg)
    ft_model.train(dataloader, tokenizer_model)


def benchmark(dataset_name: str) -> None:
    """Benchmark"""
    # Cuda setup
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    print(f"Benchmarking fine-tuned model with using dataset {dataset_name}")

    # Input
    cfg = FinetuningModelConfig(dataset_name=dataset_name)

    # Model
    ft_model = FinetuningModel(cfg)
    peft_net = ft_model.load_custom_adapter_model()

    # Tokenizer
    tokenizer_model = Tokenizer("tokenizer.model")

    # Benchmark
    if dataset_name == "news":
        test_fpb(model=peft_net, tokenizer=tokenizer_model)
    elif dataset_name == "alpaca":
        test_qna(model=peft_net, tokenizer=tokenizer_model)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")


def main(task: str, dataset_name: str) -> None:
    """API interface"""

    if task == "training":
        train(dataset_name=dataset_name)
    elif task == "test":
        benchmark(dataset_name=dataset_name)
    else:
        raise ValueError(f"Invalid task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
