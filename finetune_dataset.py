from typing import Tuple
from functools import partial

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset
import random


def data_collator(
    features: list, pretrained_max_len: int, pad_token_id: int, loss_ignore_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Form a batch of input and outputs for finetune model.
    This function will be an input of Pytorch dataloader"""
    max_len = max((len(feature["input_ids"]) + len(feature["labels"])) for feature in features)
    max_len = min(max_len, pretrained_max_len)

    input_ids = []
    padded_labels = []

    for feature in features:
        ids = feature["input_ids"][:max_len]
        labels = feature["labels"][:max_len]

        # Add padding to labels
        tmp_labels = labels + [pad_token_id] * (max_len - len(labels))
        tmp_labels = [loss_ignore_index if x == pad_token_id else x for x in tmp_labels]

        # Add padding to inputs
        ids.extend([pad_token_id] * (max_len - len(ids)))

        input_ids.append(torch.LongTensor(ids))
        padded_labels.append(torch.LongTensor(tmp_labels))

    return torch.stack(input_ids), torch.stack(padded_labels)


class FinetuningDataset(IterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        return self.generator()

    def __len__(self) -> None:
        return self.dataset.num_rows

    def generator(self) -> list:
        while True:
            self.dataset.shuffle(seed=42)
            for item in self.dataset:
                yield item


def batch_iterator(
    dataset: Dataset,
    batch_size: int,
    device: str,
    pretrained_max_len: int,
    pad_token_id: int,
    loss_ignore_index: int,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterator for mini-batch of data"""
    collate_fn = partial(
        data_collator,
        pad_token_id=pad_token_id,
        pretrained_max_len=pretrained_max_len,
        loss_ignore_index=loss_ignore_index,
    )
    ft_dataset = FinetuningDataset(dataset)
    dataloader = DataLoader(
        ft_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    for input_data, output_data in dataloader:
        input_data = input_data.to(device, non_blocking=False)
        output_data = output_data.to(device, non_blocking=False)
        yield input_data, output_data
