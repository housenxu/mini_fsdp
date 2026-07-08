import torch
from torch import nn
from torch.nn import functional as F


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
