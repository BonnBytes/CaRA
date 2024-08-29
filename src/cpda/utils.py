"""Utilities for CPDeApprox pakage."""

from typing import Optional, Tuple

import torch as th
import torch.nn as nn


class LinearToTensor:
    """Class to tensorize linear layers in attention."""

    def __init__(self, num_heads: int) -> None:
        """Init module.

        Args:
            num_heads (int): Number of heads
        """
        self.num_heads = num_heads

    def parameterize(
        self, weight: th.Tensor, is_bias: Optional[bool] = False
    ) -> nn.Parameter:
        """Convert weight and bias to tensor.

        Args:
            weight (th.Tensor): Layer weight or bias with respective shapes [dim1, dim2] and [dim1].
            is_bias (bool, optional): Boolean to specify when bias is tensorized. Defaults to False.

        Returns:
            nn.Parameter: Tensorized weight or bias of shapes [heads, dim1//heads, dim2] and [heads, dim1//heads]
        """
        embed_dim1 = weight.shape[0]
        head_dim = embed_dim1 // self.num_heads
        new_shape = (
            (self.num_heads, head_dim, weight.shape[1])
            if not is_bias
            else (self.num_heads, head_dim)
        )
        weight_tensor = weight.reshape(new_shape)
        weight_tensor = weight_tensor.permute(2, 0, 1) if not is_bias else weight_tensor
        return nn.Parameter(weight_tensor, requires_grad=weight.requires_grad)

    def __extract_weights(self, layer: nn.Module) -> Tuple[th.Tensor, ...]:
        """Extract linear layer weight and bias.

        Args:
            layer (nn.Linear): Linear layer.

        Raises:
            RuntimeError: Error if layer is not linear.

        Returns:
            Tuple[th.Tensor]: Tuple containing weight and bias tensor.
        """
        if not isinstance(layer, nn.Linear):
            raise RuntimeError("Tensorization of only linear layers is supported.")
        layer_weight = layer.weight
        layer_bias = layer.bias
        # Define a zero constant bias if not defined
        if layer_bias is None:
            layer_bias = nn.Parameter(
                th.zeros((layer_weight.shape[0])), requires_grad=False
            )
        return layer_weight, layer_bias

    def tensorize(self, layer: nn.Module) -> Tuple[nn.Parameter, ...]:
        """Tensorize a linear layer.

        Args:
            layer (nn.Module): Linear layer.

        Raises:
            RuntimeError: Error if linear out and in dimensions are not same.

        Returns:
            Tuple[nn.Parameter, ...]: Tuple containing tensorized weight and bias.
        """
        layer_weight, layer_bias = self.__extract_weights(layer)
        dim1, dim2 = layer_weight.shape
        if dim1 != dim2:
            raise RuntimeError(
                "Tensorization only supported for linear layers with equal out and in dimensions."
            )
        tensorized_weight = self.parameterize(layer_weight)
        tensorized_bias = self.parameterize(layer_bias, True)
        return tensorized_weight, tensorized_bias

    def merged_tensorize(self, layer: nn.Module) -> Tuple[nn.Parameter, ...]:
        """Tensorize a merged linear implementation.

        Args:
            layer (nn.Module): Merged linear layer.

        Raises:
            RuntimeError: Error if linear out dim not 3 times in dim.

        Returns:
            Tuple[nn.Parameter, ...]: Tuple of tuple containing Q,K,V weight and bias.
        """
        layer_weight, layer_bias = self.__extract_weights(layer)
        weight_grad = layer_weight.requires_grad
        bias_grad = layer_bias.requires_grad
        dim1, dim2 = layer_weight.shape
        if dim1 != 3 * dim2:
            raise RuntimeError(
                "Merged tensorize only support merged QKV projection implementation."
            )
        dim1 //= 3
        layer_weight = layer_weight.reshape(3, dim1, dim2)
        layer_bias = (
            layer_bias.reshape(3, dim1) if layer_bias is not None else layer_bias
        )
        Wq, Wk, Wv = layer_weight.unbind(0)
        Bq, Bk, Bv = layer_bias.unbind(0)
        Wq, Wk, Wv = (
            self.parameterize(Wq),
            self.parameterize(Wk),
            self.parameterize(Wv),
        )
        Bq, Bk, Bv = (
            self.parameterize(Bq, True),
            self.parameterize(Bk, True),
            self.parameterize(Bv, True),
        )
        weight_tensors = nn.Parameter(
            th.stack((Wq, Wk, Wv), dim=0), requires_grad=weight_grad
        )
        bias_tensors = nn.Parameter(
            th.stack((Bq, Bk, Bv), dim=0), requires_grad=bias_grad
        )
        return weight_tensors, bias_tensors
