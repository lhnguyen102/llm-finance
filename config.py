from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class NetworkConfig:
    """Network properties for Llama 7B model"""

    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = 6
    vocab_size: int = 32_000
    hidden_size: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 128
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**cfg)


@dataclass
class ModelConfig:
    """Model configuration"""

    out_dir: str = "out"
    eval_interval: int = 500
    log_interval: int = 1
    eval_iters: int = 100
    eval_only: bool = False
    alway_save_checkpoint: bool = False
    init_from: str = "scratch"

    batch_size: int = 64
    vocab_source: str = "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
    gradient_accumulation_steps: float = 4
    learning_rate: float = 5e-4
    min_lr: float = 0.0
    max_iters: int = 100000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 1000
    device: str = "cuda"
    dtype: int = "float16"
    compile: bool = True
    ddp: bool = False
    lr_decay_iters: int = max_iters

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(**cfg)