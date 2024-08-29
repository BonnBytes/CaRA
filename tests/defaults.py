"""Default values for testing."""

from dataclasses import dataclass
from typing import Optional

import torch as th
import torch.nn as nn

th.set_default_dtype(th.float64)

from src.cpda.cpda import CPDeApprox
from src.cpda.nn import CPLinear
from src.cpda.utils import LinearToTensor


@dataclass
class Defaults:
    """Default values used in testing."""

    attn_input: th.Tensor = th.randn((3, 5, 768))
    batch_size: int = 3
    cpd_input: th.Tensor = th.randn((20, 5, 4))
    cpd_rank: int = 20
    cpd_object: CPDeApprox = CPDeApprox(
        rank=20, device=th.device("cpu"), num_heads=12, random_state=10
    )
    embed_dim: int = 768
    num_heads: int = 12
    tensorize_fn: LinearToTensor = LinearToTensor(num_heads=12)
    seq_len: int = 5


class Dummy(nn.Module):
    """Dummy model."""

    def __init__(self, cpda: Optional[bool] = False):
        """Model initialization.

        Args:
            cpda (bool, optional): Booleen to use CP layer or not. Defaults to False.
        """
        super().__init__()
        self.save_path = "./test_saved"
        self.layers = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(20, 768),
        )
        self.norm = nn.GroupNorm(768, 768)
        self.final = CPLinear(768, 768) if cpda else nn.Linear(768, 768)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Dummy output.
        """
        return self.final(self.norm(self.layers(x)))
