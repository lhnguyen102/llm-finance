from model import LLAMAModel
from config import NetworkConfig, ModelConfig
from functools import partial
from tinystories import Task
import torch
import os


def main():
    """Training API"""
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # Get config
    cfg_net = NetworkConfig()
    cfg_model = ModelConfig()
    os.makedirs(cfg_model.out_dir, exist_ok=True)

    # Task-specific setup
    dataloader = partial(
        Task.iter_batches,
        batch_size=cfg_model.batch_size,
        max_seq_len=cfg_net.max_seq_len,
        vocab_size=cfg_net.vocab_size,
        vocab_source=cfg_model.vocab_source,
        device=cfg_model.device,
        num_workers=0,
    )

    # Model
    model = LLAMAModel(cfg_model=cfg_model, cfg_net=cfg_net)
    model.train(dataloader)
    # model.sample()


if __name__ == "__main__":
    main()
