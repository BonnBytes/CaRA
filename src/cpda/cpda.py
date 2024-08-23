"""CPDecomposed LoRA code."""

import warnings
from functools import reduce
from typing import Any, Dict, Optional, Tuple

import tensorly as tl
import torch as th
import torch.nn as nn

from src.cpda.utils import LinearToTensor

tl.set_backend("pytorch")


class CPDeApprox:
    """Decomposition class."""

    def __init__(
        self,
        rank: int,
        device: th.device,
        num_heads: int,
        random_state: Optional[int] = 10,
    ) -> None:
        """Init module.

        Args:
            rank (int): Rank hyperparameter.
            device (th.device): Device to use for decomposition.
            num_heads (int): Number of heads in transformer layer.
            random_state (int, optional): Randomstate for CP Decomposition. Defaults to 10.
        """
        self.rank = rank
        self.decomposition = tl.decomposition.CP(
            rank=self.rank,
            normalize_factors=True,
            verbose=False,
            init="svd",
            tol=1e-24,
            random_state=random_state,
        )
        self.device = device
        self.tensorize = LinearToTensor(num_heads=num_heads)

    @staticmethod
    def tensor_forward(
        layer_params: Tuple[th.Tensor, ...],
        x: th.Tensor,
        merged: Optional[bool] = False,
    ) -> th.Tensor:
        """Einsum based linear layer forward pass.

        Args:
            layer_params (Tuple[th.Tensor, ...]): Tuple containing weight and bias.
            x (th.Tensor): Input tensor.
            merged (bool, optional): Boolean specifying if merged QKV implementation. Defaults to False.

        Returns:
            th.Tensor: Forward pass output tensor.
        """
        weight, bias = layer_params
        if not merged:
            matmul_add = th.einsum("bse, ehd -> bshd", x, weight) + bias.unsqueeze(
                0
            ).unsqueeze(0)
            return matmul_add.permute(0, 2, 1, 3)
        else:
            matmul_add = th.einsum("bse, tehd -> tbshd", x, weight) + bias.unsqueeze(
                1
            ).unsqueeze(1)
            return matmul_add.permute(0, 1, 3, 2, 4)

    def decompose(
        self, weight: th.Tensor, sort: Optional[bool] = True
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """CP-Decompose tensor and sort based on lambdas.

        Args:
            weight (th.Tensor): Weight tensor.
            sort (Optional[bool], optional): Boolean to condition lambda sorting. Defaults to True.

        Returns:
            Tuple[th.Tensor, Tuple[th.Tensor, ...]]: Tuple containing lambdas and factors
        """
        lambdas, factors = self.decomposition.fit_transform(weight.to(self.device))
        # Move back to CPU later to save memory on GPU
        lambdas = lambdas.to("cpu")
        factors = (mat.to("cpu") for mat in factors)
        if sort:
            lambdas, factors = self.sort_factors(lambdas, factors)
        return lambdas, factors

    def sort_factors(
        self, lambdas: th.Tensor, factors: Tuple[th.Tensor, ...]
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """Sort CP factors based on lambdas.

        Args:
            lambdas (th.Tensor): Vectorized tensor containing lambdas.
            factors (Tuple[th.Tensor]): Tuple containing factor matrices.

        Returns:
            Tuple[th.Tensor, Tuple[th.Tensor, ...]]: Tuple containig sorted lambdas and factors.
        """
        sorted_val, sorted_idx = th.sort(lambdas, descending=True)
        sorted_factors = []
        for matrix in factors:
            sorted_factors.append(matrix[:, sorted_idx])
        return sorted_val, tuple(sorted_factors)

    def split_factors(
        self, lambdas: th.Tensor, factors: Tuple[th.Tensor, ...], trainable_factors: int
    ) -> Tuple[nn.Parameter, ...]:
        """Split the CPD factors into pretrained and finetuned weights.

        Args:
            lambdas (th.Tensor): Sorted lambdas.
            factors (Tuple[th.Tensor, ...]): Sorted factors.
            trainable_factors (int): Number of trainable rank-1 vectors.

        Returns:
            Tuple[nn.Parameter, ...]: Tuple containing pretrained and finetuned weights.
        """
        lambdas_pt, lambdas_ft = (
            lambdas[:-trainable_factors],
            lambdas[-trainable_factors:],
        )
        factors_pt = tuple(matrix[:, :-trainable_factors] for matrix in factors)
        factors_ft = tuple(matrix[:, -trainable_factors:] for matrix in factors)
        tensor_pt = th.nn.Parameter(
            tl.cp_to_tensor((lambdas_pt, factors_pt)), requires_grad=False
        )
        # Finetune tensor
        tensor_ft = th.nn.Parameter(
            tl.cp_to_tensor((lambdas_ft, factors_ft)), requires_grad=True
        )
        return tensor_pt, tensor_ft

    def __get_module_by_name(self, module: nn.Module, name: str) -> Any:
        """Return module type by name.

        Args:
            module (nn.Module): Model.
            name (str): layer name.

        Returns:
            Any: Attribute of the model for given name.
        """
        names = name.split(".")
        return reduce(getattr, names, module)

    def collect_cplayer_names(self, module: nn.Module) -> Dict[str, Any]:
        """Get all CPlayers.

        Args:
            module (nn.Module): Model.

        Returns:
            Dict[str, Any]: Dictionary containing layer name and layer.
        """
        layer_names = {}
        for param in module.state_dict():
            params = param.split(".")
            param = ".".join(params[:-1])
            if param not in layer_names:
                lyr = self.__get_module_by_name(module, param)
                layer_name = str(lyr).split("(")[0]
                if "CP" in layer_name:
                    layer_names[param] = lyr
        return layer_names

    def convert_state_dict(
        self, model: nn.Module, state_dict: Any, only_load: Optional[bool] = False
    ) -> nn.Module:
        """Compute CPDecomposition and initialize CPlayers.

        Args:
            model (nn.Module): Model.
            state_dict (Any): Model's state dict.
            only_load (bool, optional): Boolean to execute only loading without CP conversion. Defaults to False.

        Returns:
            nn.Module: Weight loaded model.
        """
        if only_load:
            # Only load the model without conversion
            model.load_state_dict(state_dict)
            return model
        layer_names = self.collect_cplayer_names(model)
        model.load_state_dict(state_dict, strict=False)

        for _, lyr in layer_names.items():
            if not lyr.merged_impl:
                weight_tensor, weight_bias = self.tensorize.tensorize(lyr)
                pt_weight, ft_weight = self.__process_weights(
                    weight_tensor, lyr.trainable_factors
                )
            else:
                # TODO: Dirty code: This is naive decomposition of each layer.
                weight_tensor, weight_bias = self.tensorize.merged_tensorize(lyr)
                pt_list, ft_list = [], []
                for idx in range(3):
                    p, f = self.__process_weights(
                        weight_tensor[idx], lyr.trainable_factors
                    )
                    pt_list.append(p)
                    ft_list.append(f)
                pt_weight = nn.Parameter(th.stack(pt_list), requires_grad=False)
                ft_weight = nn.Parameter(th.stack(ft_list), requires_grad=True)
            lyr.weight = pt_weight
            lyr.ft_weight = ft_weight
            lyr.bias = weight_bias
        return model

    def __process_weights(self, weight: th.Tensor, tf: int) -> Tuple[th.Tensor, ...]:
        lambdas, factors = self.decompose(weight)
        pt_weight, ft_weight = self.split_factors(
            lambdas=lambdas, factors=factors, trainable_factors=tf
        )
        return pt_weight, ft_weight

    def enable_grad(self, model: nn.Module) -> None:
        """Enable gradients only for CP added layers.

        Warnings if there is no CP layers.
        Args:
            model (nn.Module): Model.
        """
        layer_names = self.collect_cplayer_names(model)
        is_cp = len(list(layer_names.keys())) > 0
        if not is_cp:
            warnings.warn(
                "No CP layers are available to enable gradient for finetuned layers.",
                stacklevel=2,
            )
            return None
        # Turn off gradients for whole network
        for param in model.parameters():
            param.requires_grad = False
        # Enable gradients only for FT weights
        for name, param in model.named_parameters():
            if "ft_weight" in name:
                param.requires_grad_(True)
        return None
