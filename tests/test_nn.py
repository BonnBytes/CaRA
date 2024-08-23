"""Test CP formed layers."""

from typing import Optional

import pytest
import tensorly as tl
import torch as th

tl.set_backend("pytorch")

from src.cpda.cpda import CPDeApprox
from src.cpda.nn import CPLinear
from tests.defaults import Defaults
from tests.test_cpda import _create_delete_folder

th.set_default_dtype(th.float64)


class SingleLayer(th.nn.Module):
    """Single layer small network."""

    def __init__(
        self,
        in_dims: int,
        bias: bool,
        merged: Optional[bool] = False,
        cpda: Optional[bool] = False,
    ) -> None:
        """Model initialization.

        Args:
            in_dims (int): Input features.
            bias (bool): Boolean to use bias.
            merged (bool, optional): Boolean to use merged QKV implementation. Defaults to False.
            cpda (bool, optional): Boolean to use CP layers or not. Defaults to False.
        """
        super().__init__()
        out_dims = in_dims if not merged else in_dims * 3
        self.final = (
            CPLinear(in_dims, out_dims, bias)
            if cpda
            else th.nn.Linear(in_dims, out_dims, bias=bias)
        )

    def forward(self, x):
        """Forward pass."""
        return self.final(x)


def _local_init():
    return Defaults(
        attn_input=th.randn((3, 5, 16)),
        embed_dim=16,
        num_heads=4,
        cpd_rank=16,
        cpd_object=CPDeApprox(
            rank=16, device=th.device("cpu"), num_heads=4, random_state=10
        ),
    )


@pytest.mark.parametrize("bias", [False, True])
def test_layer(bias: bool):
    """Test CPLinear layer implementation.

    Args:
        bias (bool): Boolean to use bias.
    """
    DFlt_local = _local_init()
    head_dim = DFlt_local.embed_dim // DFlt_local.num_heads

    model = SingleLayer(DFlt_local.embed_dim, bias, merged=False, cpda=False)
    _create_delete_folder(model, "./test_saved/", create=True)
    model_out = (
        model(DFlt_local.attn_input)
        .reshape(
            DFlt_local.batch_size, DFlt_local.seq_len, DFlt_local.num_heads, head_dim
        )
        .permute(0, 2, 1, 3)
    )

    cpmodel = SingleLayer(DFlt_local.embed_dim, bias, merged=False, cpda=True)
    old_state_dict = th.load("./test_saved/dummy.pt")
    _create_delete_folder(model, "./test_saved", create=False)
    cpmodel = DFlt_local.cpd_object.convert_state_dict(cpmodel, old_state_dict)
    cp_out = cpmodel(DFlt_local.attn_input)
    del DFlt_local
    assert th.allclose(cp_out, model_out)


@pytest.mark.parametrize("bias", [False, True])
def test_merged_layer(bias: bool):
    """Test CPLinear layer with merged QKV implementation.

    Args:
        bias (bool): Boolean to use bias.
    """
    DFlt_local = _local_init()
    head_dim = DFlt_local.embed_dim // DFlt_local.num_heads

    model = SingleLayer(DFlt_local.embed_dim, bias, merged=True, cpda=False)
    _create_delete_folder(model, "./test_saved/", create=True)
    model_out = (
        model(DFlt_local.attn_input)
        .reshape(
            DFlt_local.batch_size, DFlt_local.seq_len, 3, DFlt_local.num_heads, head_dim
        )
        .permute(2, 0, 3, 1, 4)
    )

    cpmodel = SingleLayer(DFlt_local.embed_dim, bias, merged=True, cpda=True)
    old_state_dict = th.load("./test_saved/dummy.pt")
    _create_delete_folder(model, "./test_saved", create=False)
    cpmodel = DFlt_local.cpd_object.convert_state_dict(cpmodel, old_state_dict)
    cp_out = cpmodel(DFlt_local.attn_input)
    del DFlt_local
    assert th.allclose(cp_out, model_out)
