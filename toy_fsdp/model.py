from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from .profiling import trace_range


class TinyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(x)
        if y is None:
            return logits
        return F.cross_entropy(logits, y)


def make_batch(
    batch_size: int,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(batch_size, input_dim, generator=generator, device=device)
    w = torch.randn(input_dim, num_classes, generator=generator, device=device)
    y = (x @ w).argmax(dim=-1)
    return x, y


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    max_seq_len: int = 128
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_dim: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    freeze_token_embedding: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.max_seq_len <= 1:
            raise ValueError("vocab_size must be positive and max_seq_len must exceed one")
        if self.hidden_dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_dim, num_layers, and num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.dropout = config.dropout
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=True)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            hidden_dim,
        )
        return self.out_proj(attention)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_dim,
            2 * config.intermediate_dim,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_dim,
            config.hidden_dim,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    """A pre-norm causal Transformer block and one FSDP2 communication unit."""

    def __init__(self, config: TransformerConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_dim)
        self.mlp = SwiGLUMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with trace_range(f"transformer::block_{self.layer_id}"):
            hidden_states = hidden_states + self.attention(
                self.attention_norm(hidden_states)
            )
            hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
            return hidden_states


class TinyTransformerLM(nn.Module):
    """Small causal LM designed to exercise realistic FSDP2 parameter cases."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if config.freeze_token_embedding:
            self.token_embedding.weight.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, seq_len = input_ids.shape
        del batch_size
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))
        if labels is None:
            return logits
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
            labels[:, 1:].contiguous().view(-1),
        )


def make_lm_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
        device=device,
    )
    return input_ids, input_ids.clone()
