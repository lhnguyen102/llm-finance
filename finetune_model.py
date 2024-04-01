import inspect
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, Iterator, List, Tuple, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import parametrize

from config import FinetuningModelConfig, NetworkConfig
from model import LLAMANet
from tinystories import get_tokenizer_model_path
from tokenizer import Tokenizer


class LoraLinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models - https://arxiv.org/abs/2106.09685

    Heavily inspired by minLoRA - https://github.com/cccntu/minLoRA

    Leverages the parametrizations feature from pytorch. This allows us to add the LoRA
    matrices to the weights during the forward pass rather than computing the modified
    forward pass explicitly, i.e., we compute (W + BA)x rather than Wx + BAx.
    """

    def __init__(self, fan_in, fan_out, rank=4, dropout_p=0.0, alpha=1.0):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.rank = rank
        self.dropout_p = dropout_p

        self.lora_a = nn.Parameter(torch.zeros(rank, fan_in))
        self.lora_b = nn.Parameter(torch.zeros(fan_out, rank))

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, weight):
        return (
            weight + torch.matmul(self.lora_b, self.dropout(self.lora_a)) * self.scaling
        )


def apply_lora(
    model: nn.Module, layer_types=[nn.Linear], rank=8, dropout=0.0, alpha=1.0
):
    def _apply_lora(module):
        if type(module) in layer_types and hasattr(module, "weight"):
            fan_out, fan_in = module.weight.shape
            parametrize.register_parametrization(
                module, "weight", LoraLinear(fan_in, fan_out, rank, dropout, alpha)
            )

    model.apply(_apply_lora)


def tie_lora_weights(src, trg):
    """Tie the LoRA weights between two modules. Can be useful for tying embeddings to the final classifier."""
    if hasattr(src, "parametrizations") and hasattr(trg, "parametrizations"):
        trg.parametrizations.weight[0].lora_a = src.parametrizations.weight[0].lora_a
        trg.parametrizations.weight[0].lora_b = src.parametrizations.weight[0].lora_b


class FinetuningModel:
    """Custom finetune trainer"""

    def __init__(self, cfg: FinetuningModelConfig) -> None:
        self.cfg = cfg
        device_type = "cuda" if "cuda" in cfg.device else "cpu"
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.dtype]
        self.ctx = (
            nullcontext()
            if cfg.device == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # self.peft_net = self.get_lora_finetune_model()
        self.peft_net = self.get_custom_lora_finetune_model()

    def train(self, dataloader: Dict[str, Iterator], tokenizer) -> None:
        """Finetune the model"""

        # Get optimizer
        optimizer = self._get_optimizer()

        # Fetch the very first batch
        input_ids, labels = next(dataloader["train"])

        # Initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.dtype == "float16"))

        local_iter_num = 0
        iter_num = 0
        best_val_loss = 1e9
        time_start = time.perf_counter()
        while True:
            # Validation step
            if iter_num % self.cfg.eval_interval == 0:
                print("Validating...")
                losses = self.validate(dataloader=dataloader)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if losses["val"] < best_val_loss or self.cfg.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        self.save_adapter_model()
                        # self.save_model()

            # Learning rate decaying
            lr = self.get_lr(iter_num) if self.cfg.decay_lr else self.cfg.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            for _ in range(self.cfg.gradient_accumulation_steps):
                with self.ctx:
                    _ = self.peft_net(input_ids, labels)

                    loss = self.peft_net.last_loss
                    loss = loss / self.cfg.gradient_accumulation_steps

                # Immediately async prefetch next batch while model is doing the forward pass on the GPU
                input_ids, labels = next(dataloader["train"])

                # Backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # Clip the gradient
            if self.cfg.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.peft_net.parameters(), self.cfg.grad_clip
                )

            # Step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Loggings
            time_end = time.perf_counter()
            dt = time_end - time_start
            time_start = time_end

            if iter_num % self.cfg.log_interval == 0:
                # Get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                lossf = loss.item() * self.cfg.gradient_accumulation_steps
                print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms")

            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.cfg.max_iters:
                break

    @torch.no_grad()
    def validate(self, dataloader: Dict[str, Iterator]) -> Tuple[float, float]:
        """Validate model"""
        out = {}
        self.peft_net.eval()
        for split, batch_iter in dataloader.items():
            losses = torch.zeros(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                input_ids, labels = next(batch_iter)
                with self.ctx:
                    _ = self.peft_net(input_ids, labels)
                    loss = self.peft_net.last_loss
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.peft_net.train()

        return out

    def sample(self) -> None:
        """Generate texts from trained model"""
        # Load the model
        peft_net = self.load_custom_adapter_model()
        start = ""

        # Load the tokenizer
        vocab_source = self.cfg.vocab_source
        vocab_size = peft_net.cfg.vocab_size

        # Let's try to find the tokenizer model automatically
        query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
        tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
        enc = Tokenizer(tokenizer_model=tokenizer_model)

        # Encode the beginning of the prompt
        if start.startswith("FILE:"):
            with open(start[5:], "r", encoding="utf-8") as f:
                start = f.read()
        start_ids = enc.encode(start, bos=True, eos=False)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.cfg.device)[None, ...]

        # Run generation
        with torch.no_grad():
            with self.ctx:
                for k in range(1):
                    y = peft_net.generate(x, 100, temperature=1.0, top_k=300)
                    print(enc.decode(y[0].tolist()))
                    print("---------------")

    def get_sentiment_analysis(
        self, prompts: Union[List[int], List[str]]
    ) -> Dict[str, float]:
        """Get the probabilities for the following sentiment: positive, negative, and neural"""

        # Load the model
        peft_net = self.load_custom_adapter_model()

        # Load the tokenizer
        vocab_source = self.cfg.vocab_source
        vocab_size = peft_net.cfg.vocab_size

        # Let's try to find the tokenizer model automatically
        query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
        tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
        enc = Tokenizer(tokenizer_model=tokenizer_model)

        # Get sentiment ids
        sentiments = ["positive", "negative", "neural"]
        sentiment_ids = {}
        for name in sentiments:
            sentiment_ids[name] = tokenizer_model.encode(name)

        # Encode the beginning of the prompt
        if all(isinstance(item, str) for item in prompts):
            prompt_ids = []
            for prompt in prompts:
                prompt_id = enc.encode(prompt, bos=True, eos=False)
                prompt_ids.append(prompt_id)
            prompt_ids = torch.tensor(
                prompt_ids, dtype=torch.long, device=self.cfg.device
            )[None, ...]
        else:
            prompt_ids = torch.tensor(
                prompts, dtype=torch.long, device=self.cfg.device
            )[None, ...]

        # Run generation
        with torch.no_grad():
            with self.ctx:
                token_probs = peft_net.compute_probs(
                    prompt_ids, temperature=1.0, top_k=300
                )

        # Get probabilities for each type of sentiment
        sentiment_probs = {}
        for name, ids in sentiment_ids.items():
            prob = 0
            for i in ids:
                prob += token_probs[i]
            sentiment_probs[name] = prob

        # Normalize the probabilities for each type of sentiment
        tot_prob = sum(sentiment_probs.values())
        for name, val in sentiment_probs:
            sentiment_probs[name] = val / tot_prob

        return sentiment_probs

    def save_adapter_model(self):
        """Save model adapter including the parameters optimized for the finetuning"""
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        saved_params = {
            k: v.to("cpu")
            for k, v in self.peft_net.named_parameters()
            if v.requires_grad
        }
        torch.save(
            saved_params,
            os.path.join(
                self.cfg.out_dir, f"adapter_model_{self.cfg.dataset_name}.bin"
            ),
        )

    def get_lora_finetune_model(self) -> PeftModel:
        """Get LoRA finetuning model using Huggingface's library"""
        # Load pretrained model
        pretrained_net = self.load_pretrained_network()

        # Pretrained model
        target_modules = [
            n
            for n, _ in pretrained_net.named_modules()
            if any(x in n for x in ("query", "value", "fc"))
        ]

        # LoRA config
        lora_config = LoraConfig(
            inference_mode=False,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        peft_net = get_peft_model(pretrained_net, lora_config)
        self.print_trainable_parameters(peft_net)

        return peft_net

    def load_adapter_model(self) -> None:
        """Load adapter model using Huggingface's library"""
        # Load pretrain model
        pretrained_net = self.load_pretrained_network()

        # Pretrained model
        target_modules = [
            n
            for n, _ in pretrained_net.named_modules()
            if any(x in n for x in ("query", "value", "fc"))
        ]

        # LoRA config
        lora_config = LoraConfig(
            inference_mode=False,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        peft_net = get_peft_model(pretrained_net, lora_config)
        peft_saved_params = torch.load(
            os.path.join(
                self.cfg.out_dir, f"adapter_model_{self.cfg.dataset_name}.bin"
            ),
            map_location=self.cfg.device,
        )

        # Load adapter parameter
        with torch.no_grad():
            for name, param in peft_net.named_parameters():
                if name in peft_saved_params:
                    param.copy_(peft_saved_params[name])

        return peft_net

    def load_custom_adapter_model(self) -> None:
        """Load adapter model using LoRA's custom implementation"""
        # Load pretrain model
        pretrained_net = self.load_pretrained_network()

        # Load checkpoint
        peft_saved_params = torch.load(
            os.path.join(
                self.cfg.out_dir, f"adapter_model_{self.cfg.dataset_name}.bin"
            ),
            map_location=self.cfg.device,
        )

        # Apply LoRA
        for p in pretrained_net.parameters():
            p.requires_grad = False

        apply_lora(
            pretrained_net,
            layer_types=[nn.Linear, nn.Embedding],
            rank=self.cfg.lora_rank,
            dropout=self.cfg.lora_dropout,
            alpha=self.cfg.lora_alpha,
        )

        # Set embedding's weight to be equal to the output layer
        if self.cfg.lora_tie_embedding_weights:
            tie_lora_weights(pretrained_net.output, pretrained_net.tok_embeddings)

        # Load adapter parameter
        with torch.no_grad():
            for name, param in pretrained_net.named_parameters():
                if name in peft_saved_params:
                    param.copy_(peft_saved_params[name])

        return pretrained_net.to(self.cfg.device)

    def get_custom_lora_finetune_model(self) -> LLAMANet:
        """Get the finetune model using LoRA's custom implementation"""
        # Load pretrained model
        pretrained_net = self.load_pretrained_network()

        # Apply LoRA
        for p in pretrained_net.parameters():
            p.requires_grad = False

        apply_lora(
            pretrained_net,
            layer_types=[nn.Linear, nn.Embedding],
            rank=self.cfg.lora_rank,
            dropout=self.cfg.lora_dropout,
            alpha=self.cfg.lora_alpha,
        )

        # Set embedding's weight to be equal to the output layer
        if self.cfg.lora_tie_embedding_weights:
            tie_lora_weights(pretrained_net.output, pretrained_net.tok_embeddings)

        pretrained_net.to(self.cfg.device)

        return pretrained_net

    def load_pretrained_network(self, ckpt_path: str = "./out/ckpt.pt") -> LLAMANet:
        """Load model checkpoint"""
        checkpoint = torch.load(ckpt_path, map_location=self.cfg.device)
        checkpoint_model_args = checkpoint["cfg_net"]
        state_dict = checkpoint["model"]

        # Create the model
        cfg_net = NetworkConfig()
        cfg_net.from_dict(checkpoint_model_args)
        llama_net = LLAMANet(cfg_net)
        llama_net.load_state_dict(state_dict)

        # llama_net.to(self.cfg.device)

        return llama_net

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Define the optimizers"""
        # Get all parameters
        param_dict = {
            name: value
            for name, value in self.peft_net.named_parameters()
            if value.requires_grad
        }

        # Select the ones to be decayed
        decay_params = [value for _, value in param_dict.items() if value.dim() >= 2]
        nodecay_params = [value for _, value in param_dict.items() if value.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
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
        if self.cfg.optim_method == "8bit":
            optimizer = bnb.optim.AdamW8bit(
                optim_groups,
                lr=self.cfg.learning_rate,
                betas=(self.cfg.beta1, self.cfg.beta2),
            )
        else:
            fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_avail and self.cfg.device == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.cfg.learning_rate,
                betas=(self.cfg.beta1, self.cfg.beta2),
                **extra_args,
            )

        return optimizer

    def get_lr(self, it: int):
        # 1) linear warmup for warmup_iters steps
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * it / self.cfg.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.cfg.lr_decay_iters:
            return self.cfg.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.cfg.warmup_iters) / (
            self.cfg.lr_decay_iters - self.cfg.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

        return self.cfg.min_lr + coeff * (self.cfg.learning_rate - self.cfg.min_lr)

    @staticmethod
    def print_trainable_parameters(model: PeftModel):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
