import inspect
import math
import os
import struct
import time
from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn

from config import ModelConfig, NetworkConfig
from tinystories import get_tokenizer_model_path
from tokenizer import Tokenizer


class RMSNorm(nn.Module):
    """Running mean normalization"""

    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        self.eps = cfg.norm_eps
        self.weight = nn.Parameter(torch.ones(cfg.dim))

    def _norm(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * torch.rsqrt(obs.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        output = self._norm(obs.float()).type_as(obs)
        return output * self.weight


def reshape_for_broadcast(freqs_cis: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    ndim = obs.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (obs.shape[1], obs.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(obs.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    query: torch.Tensor, key: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Positional embeddings"""

    # Reshape query and key to match the complex representation
    query_r, query_i = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_r, key_i = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, query_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, query_r)

    # Apply rotation using real numbers
    query_out_r = query_r * freqs_cos - query_i * freqs_sin
    query_out_i = query_r * freqs_sin + query_i * freqs_cos
    key_out_r = key_r * freqs_cos - key_i * freqs_sin
    key_out_i = key_r * freqs_sin + key_i * freqs_cos

    # Flatten last two dimensions
    query_out = torch.stack([query_out_r, query_out_i], dim=-1).flatten(3)
    key_out = torch.stack([key_out_r, key_out_i], dim=-1).flatten(3)

    return query_out.type_as(query), key_out.type_as(query)


def repeat_kv(obs: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat tensor element"""
    batch_size, seq_len, n_kv_heads, head_dim = obs.shape
    if n_rep == 1:
        return obs
    return (
        obs[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    timestep = torch.arange(end, device=freqs.device)
    freqs = torch.outer(timestep, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


class Attention(nn.Module):
    """Attention mechanism"""

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.n_kv_heads = cfg.n_heads if cfg.n_kv_heads is None else cfg.n_kv_heads
        assert cfg.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = cfg.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.query = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.key = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.res_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout

        self.flash = hasattr(nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attn. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, cfg.max_seq_len, cfg.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self, obs: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = obs.shape

        # QKV
        query, key, value = self.query(obs), self.key(obs), self.value(obs)
        query = query.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        # Relative positional embeddings
        query, key = apply_rotary_emb(
            query=query, key=key, freqs_cos=freqs_cos, freqs_sin=freqs_sin
        )

        # Expand key and value -> (batch_size, seq_len, n_local_heads, head_dim)
        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)

        # Reshape query, key, and value to compute attn score for each time step
        # (batch_size, n_local_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Attention (batch_size, n_local_heads, seq_len, head_dim)
        if self.flash:
            output = nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :seq_len, :seq_len]
            scores = nn.functional.softmax(scores.float(), dim=-1).type_as(query)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, value)

        # Reshape back to the FC's output (batch_size, seq_len, n_local_heads, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.output_proj(output)
        output = self.res_dropout(output)

        return output


class MLPBlock(nn.Module):
    """Multi-perceptron block in Llama"""

    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        hidden_dim = cfg.dim
        if hidden_dim is None:
            hidden_dim = 4 * cfg.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = cfg.multiple_of * ((hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of)

        self.fc_11 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.fc_12 = nn.Linear(cfg.dim, hidden_dim, bias=False)

        self.fc_2 = nn.Linear(hidden_dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = nn.functional.silu(self.fc_11(obs) * self.fc_12(obs))
        out = self.fc_2(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block"""

    def __init__(self, layer_id: int, cfg: NetworkConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.dim = cfg.dim
        self.head_dim = cfg.dim // cfg.n_heads
        self.attn = Attention(cfg)
        self.mlp = MLPBlock(cfg)
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(cfg)
        self.mlp_norm = RMSNorm(cfg)

    def forward(
        self, obs: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ) -> torch.Tensor:
        hidden = obs + self.attn(obs=self.attn_norm(obs), freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        out = hidden + self.mlp(self.mlp_norm(hidden))

        return out


class LLAMANet(nn.Module):
    """Llama network"""

    last_loss: Optional[torch.Tensor]

    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=32000)
        self.dropout = nn.Dropout(cfg.dropout)
        self.transformer_layers = torch.nn.ModuleList()

        for layer_id in range(cfg.n_layers):
            self.transformer_layers.append(TransformerBlock(layer_id, cfg))

        self.norm = RMSNorm(cfg)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Share the unembedding parameters with the embedding parameters
        # https://paperswithcode.com/method/weight-tying
        self.tok_embeddings.weight = self.output.weight

        # Positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initalize the parameters
        self.apply(self._init_params)

        # Register hook to freeze the gradient for padding_idx
        if self.tok_embeddings.padding_idx is not None:
            # self.output.weight.register_hook(lambda grad: self.gradient_mask_hook(grad))
            self.padding_idx = self.tok_embeddings.padding_idx
            with torch.no_grad():
                self.tok_embeddings.weight[self.tok_embeddings.padding_idx].fill_(0.0)

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

        # Initialize attribute for the loss of the last forward call.
        self.last_loss = None

    def gradient_mask_hook(self, grad):
        grad[self.padding_idx] = 0
        return grad

    def _init_params(self, module: nn.Module) -> None:
        """Initialize weights and biases for network"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, seq_len = tokens.shape
        # self.tok_embeddings(torch.tensor([[32000]], device="cuda"))
        hidden = self.tok_embeddings(tokens)
        hidden = self.dropout(hidden)
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer in self.transformer_layers:
            hidden = layer(hidden, freqs_cos, freqs_sin)

        hidden = self.norm(hidden)

        if targets is not None:
            logits = self.output(hidden)
            self.last_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.loss_ignore_index,
            )
        else:
            logits = self.output(hidden[:, [-1], :])
            self.last_loss = None

        return logits

    @torch.inference_mode()
    def generate(
        self, idx: torch.Tensor, max_new_tokens, temperature=1.0, top_k=None
    ) -> torch.Tensor:
        """"""

        for _ in range(max_new_tokens):
            # Crop the token inputs beyond the token limits
            idx_cond = (
                idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len :]
            )

            # Get the logits for the final step only
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature

                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # apply softmax to convert logits to (normalized) probabilities
                probs = nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.inference_mode()
    def compute_probs(self, idx: torch.Tensor, temperature=1.0, top_k=None) -> torch.Tensor:
        """Compute the probabilies for each token in vocab given the input tokens"""
        # Crop the token inputs beyond the token limits
        idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len :]

        # Get the logits for the final step only
        logits = self(idx_cond)
        logits = logits[:, -1, :]

        if temperature != 0.0:
            # Apply temperature to logits in order to expand the uncertainty band
            logits = logits / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

        # Probabilities for each token in vocab
        probs = nn.functional.softmax(logits, dim=-1)

        return probs


class LLAMAModel:
    """Llama model"""

    def __init__(self, cfg_net: NetworkConfig, cfg_model: ModelConfig) -> None:
        self.cfg_net = cfg_net
        self.cfg_model = cfg_model
        self.llama_net = LLAMANet(cfg_net)
        self.llama_net.to(self.cfg_model.device)
        if cfg_model.compile:
            self.llama_net = torch.compile(self.llama_net)

        device_type = "cuda" if "cuda" in cfg_model.device else "cpu"
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            cfg_model.dtype
        ]
        self.ctx = (
            nullcontext()
            if cfg_model.device == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

    def _init_model(self) -> Tuple[int, float]:
        if self.cfg_model.init_from == "scratch":
            # Init a new model from scratch
            print("Initializing a new model from scratch")
            self.llama_net = LLAMANet(self.cfg_net)
            iter_num = 0
            best_val_loss = 1e9
            optimizer_dict = None
            self.llama_net.to(self.cfg_model.device)
        elif self.cfg_model.init_from == "resume":
            print(f"Resuming training from {self.cfg_model.out_dir}")
            # Load data from checkpoint
            ckpt_path = os.path.join(self.cfg_model.out_dir, "ckpt.pt")
            iter_num, best_val_loss, optimizer_dict = self.load_model(ckpt_path)

        return iter_num, best_val_loss, optimizer_dict

    def train(self, dataloader: Callable) -> None:
        """Traing step"""

        # Initialize model
        iter_num, best_val_loss, optimizer_dict = self._init_model()

        # Training dataloader
        train_dataloader = dataloader(split="train")

        # Get optimizer
        optimizer = self._get_optimizer()
        if optimizer_dict is not None:
            optimizer.load_state_dict(optimizer_dict)

        # Fetch the very first batch
        X, Y = next(train_dataloader)

        # Initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg_model.dtype == "float16"))

        local_iter_num = 0
        time_start = time.perf_counter()
        while True:
            if iter_num % self.cfg_model.eval_interval == 0:
                losses = self.estimate_loss(dataloader=dataloader)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

                if losses["val"] < best_val_loss or self.cfg_model.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        self.save_model(
                            optimizer=optimizer, iter_num=iter_num, best_val_loss=best_val_loss
                        )
                        self.model_export(
                            self.llama_net,
                            os.path.join(self.cfg_model.out_dir, "model.bin"),
                        )

            if iter_num == 0 and self.cfg_model.eval_only:
                break

            # Learning rate decaying
            lr = self.get_lr(iter_num) if self.cfg_model.decay_lr else self.cfg.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            for _ in range(self.cfg_model.gradient_accumulation_steps):
                with self.ctx:
                    _ = self.llama_net(X, Y)
                    loss = self.llama_net.last_loss
                    loss = loss / self.cfg_model.gradient_accumulation_steps

                # Immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = next(train_dataloader)

                # Backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # Clip the gradient
            if self.cfg_model.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.llama_net.parameters(), self.cfg_model.grad_clip
                )

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Loggings
            time_end = time.perf_counter()
            dt = time_end - time_start
            time_start = time_end

            if iter_num % self.cfg_model.log_interval == 0:
                # Get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                lossf = loss.item() * self.cfg_model.gradient_accumulation_steps
                print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms")

            iter_num += 1
            local_iter_num += 1

            # Termination conditions
            if iter_num > self.cfg_model.max_iters:
                break

    def sample(self) -> None:
        """Generate texts from trained model"""
        # Load the model
        ckpt_path = os.path.join(self.cfg_model.out_dir, "ckpt.pt")
        self.load_model(ckpt_path)
        start = ""
        self.model_export(
            model=self.llama_net, filepath=os.path.join(self.cfg_model.out_dir, "model.bin")
        )

        # Load the tokenizer
        vocab_source = self.cfg_model.vocab_source
        vocab_size = self.cfg_net.vocab_size

        # Let's try to find the tokenizer model automatically
        query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
        tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
        enc = Tokenizer(tokenizer_model=tokenizer_model)

        # Encode the beginning of the prompt
        if start.startswith("FILE:"):
            with open(start[5:], "r", encoding="utf-8") as f:
                start = f.read()
        start_ids = enc.encode(start, bos=True, eos=False)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.cfg_model.device)[None, ...]

        # Run generation
        with torch.no_grad():
            with self.ctx:
                for k in range(1):
                    y = self.llama_net.generate(x, 256, temperature=1.4, top_k=1000)
                    print(enc.decode(y[0].tolist()))
                    print("---------------")

    @torch.no_grad()
    def estimate_loss(self, dataloader: Callable):
        out = {}
        self.llama_net.eval()
        for split in ["train", "val"]:
            batch_iter = dataloader(split=split)
            losses = torch.zeros(self.cfg_model.eval_iters)  # keep on CPU
            for k in range(self.cfg_model.eval_iters):
                X, Y = next(batch_iter)
                with self.ctx:
                    logits = self.llama_net(X, Y)
                    loss = self.llama_net.last_loss
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.llama_net.train()
        return out

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Define the optimizers"""
        # Get all parameters
        param_dict = {
            name: value for name, value in self.llama_net.named_parameters() if value.requires_grad
        }

        # Select the ones to be decayed
        decay_params = [value for _, value in param_dict.items() if value.dim() >= 2]
        nodecay_params = [value for _, value in param_dict.items() if value.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg_model.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Printout stat
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Select optimizer
        optimizer = bnb.optim.Adam8bit(
            optim_groups,
            lr=self.cfg_model.learning_rate,
            betas=(self.cfg_model.beta1, self.cfg_model.beta2),
        )
        # fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_avail and self.cfg_model.device == "cuda"
        # extra_args = dict(fused=True) if use_fused else dict()
        # optimizer = torch.optim.AdamW(
        #     optim_groups,
        #     lr=self.cfg_model.learning_rate,
        #     betas=(self.cfg_model.beta1, self.cfg_model.beta2),
        #     **extra_args,
        # )
        # print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.llama_net.parameters())
        cfg = self.cfg_net
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second

        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_lr(self, it: int):
        # 1) linear warmup for warmup_iters steps
        if it < self.cfg_model.warmup_iters:
            return self.cfg_model.learning_rate * it / self.cfg_model.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.cfg_model.lr_decay_iters:
            return self.cfg_model.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.cfg_model.warmup_iters) / (
            self.cfg_model.lr_decay_iters - self.cfg_model.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

        return self.cfg_model.min_lr + coeff * (
            self.cfg_model.learning_rate - self.cfg_model.min_lr
        )

    def save_model(
        self, optimizer: torch.optim.Optimizer, iter_num: int, best_val_loss: float
    ) -> None:
        """Save model"""
        checkpoint = {
            "model": self.llama_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg_net": self.cfg_net.to_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "cfg_model": self.cfg_model.to_dict(),
        }
        print(f"saving checkpoint to {self.cfg_model.out_dir}")
        torch.save(checkpoint, os.path.join(self.cfg_model.out_dir, "ckpt.pt"))

    def load_model(self, ckpt_path: str = "./out/ckpt.pt") -> Tuple[int, float, dict]:
        """Load model checkpoint"""
        checkpoint = torch.load(ckpt_path, map_location=self.cfg_model.device)
        checkpoint_model_args = checkpoint["cfg_net"]
        state_dict = checkpoint["model"]
        optimizer_dict = checkpoint["optimizer"]
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

        # Create the model
        self.cfg_net.from_dict(checkpoint_model_args)
        self.llama_net = LLAMANet(self.cfg_net)
        self.llama_net.load_state_dict(state_dict)

        self.llama_net.to(self.cfg_model.device)

        return iter_num, best_val_loss, optimizer_dict

    @staticmethod
    def model_export(model: LLAMANet, filepath: str):
        """Original export of llama2.c bin files, i.e. version v0"""
        out_file = open(filepath, "wb")

        # first write out the header
        hidden_dim = model.transformer_layers[0].mlp.fc_11.weight.shape[0]
        p = model.cfg
        shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
        # legacy format uses negative/positive vocab size as a shared classifier flag
        if not shared_classifier:
            p.vocab_size = -p.vocab_size
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack(
            "iiiiiii",
            p.dim,
            hidden_dim,
            p.n_layers,
            p.n_heads,
            n_kv_heads,
            p.vocab_size,
            p.max_seq_len,
        )
        out_file.write(header)

        # next write out the embedding weights
        serialize_fp32(out_file, model.tok_embeddings.weight)

        # now all the layers
        # attn weights
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.attn_norm.weight)
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.attn.query.weight)
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.attn.key.weight)
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.attn.value.weight)
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.attn.output_proj.weight)
        # ffn weights
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.mlp_norm.weight)
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.mlp.fc_11.weight)
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.mlp.fc_2.weight)
        for layer in model.transformer_layers:
            serialize_fp32(out_file, layer.mlp.fc_12.weight)
        # final rmsnorm
        serialize_fp32(out_file, model.norm.weight)
        # freqs_cis
        serialize_fp32(out_file, model.freqs_cos[: p.max_seq_len])
        serialize_fp32(out_file, model.freqs_sin[: p.max_seq_len])

        # final classifier weights
        if not shared_classifier:
            serialize_fp32(out_file, model.output.weight)

        # write to binary file
        out_file.close()
        print(f"wrote {filepath}")


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)
