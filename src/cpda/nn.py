"""CPfied layers."""

from typing import Optional

import torch as th
import torch.nn as nn

import src.cpda.cpda as cpda


class CPLinear(nn.Linear):
    """Overlead Linear layer with CP form."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        device=None,
        dtype=None,
        trainable_factors: Optional[int] = 2,
    ) -> None:
        """Cplinear initialization.

        Args:
            in_features (int): Input features.
            out_features (int): Output features.
            bias (bool, optional): Boolean to use bias. Defaults to True.
            device (_type_, optional): torch device. Defaults to None.
            dtype (_type_, optional): torch dtype. Defaults to None.
            trainable_factors (int, optional): Number of trainable CP Factors. Defaults to 2.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.trainable_factors = trainable_factors
        self.merged_impl = False
        if out_features == 3 * in_features:
            self.merged_impl = True
        self.ft_weight = nn.Parameter(th.randn_like(self.weight), requires_grad=True)

    def forward(self, input_: th.Tensor) -> th.Tensor:
        """Forward pass for CPLinear layer.

        Args:
            input_ (th.Tensor): Input.

        Returns:
            th.Tensor: Tensorized attention linear output.
        """
        pt_y = cpda.CPDeApprox.tensor_forward(
            (self.weight, self.bias), input_, self.merged_impl
        )
        # TODO: Dumpster fire. Refactor tensor forward to use bias if only provided.
        ft_y = cpda.CPDeApprox.tensor_forward(
            (self.ft_weight, th.zeros_like(self.bias)), input_, self.merged_impl
        )
        return pt_y + ft_y
