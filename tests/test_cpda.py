"""Test for CPDA."""

import os
from typing import Optional

import pytest
import tensorly as tl
import torch as th

tl.set_backend("pytorch")

from src.cpda.cpda import CPDeApprox
from tests.defaults import Defaults as DFlt
from tests.defaults import Dummy

th.set_default_dtype(th.float64)


def _init_ones(module_: th.nn.Module) -> None:
    """Initialize module with ones.

    Args:
        module_ (th.nn.Module): Layer from model.
    """
    if isinstance(module_, th.nn.Linear):
        th.nn.init.ones_(module_.weight.data)
        th.nn.init.ones_(module_.bias.data)
        try:
            th.nn.init.ones_(module_.ft_weight.data)
        except Exception:
            pass


def _create_delete_folder(
    model: th.nn.Module, save_path: str, create: Optional[bool] = True
):
    """Create and delete folder for saving model.

    Args:
        model (th.nn.Module): Model.
        create (bool, optional): Boolean to specify creation or deletion. Defaults to True.
    """
    if create:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            th.save(model.state_dict(), f"{save_path}/dummy.pt")
        assert os.path.exists(save_path)
    else:
        import shutil

        shutil.rmtree(save_path)
        assert not os.path.exists(save_path)


def test_factorsort():
    """Test sorting CP factors based on lambdas."""
    lambdas, factors = DFlt.cpd_object.decomposition.fit_transform(DFlt.cpd_input)
    unsrt_recons = tl.cp_to_tensor((lambdas, factors))
    sort_l, sort_factors = DFlt.cpd_object.sort_factors(
        lambdas=lambdas, factors=factors
    )
    srt_recons = tl.cp_to_tensor((sort_l, sort_factors))
    assert not th.allclose(lambdas, sort_l)
    assert th.allclose(unsrt_recons, srt_recons)


@pytest.mark.parametrize("bias", [True, False])
def test_einsum(bias: bool):
    """Test einsum style linear layer operations.

    Args:
        bias (bool): Boolean to use bias.
    """
    head_dim: int = DFlt.embed_dim // DFlt.num_heads
    dummy_layer = th.nn.Linear(DFlt.embed_dim, DFlt.embed_dim, bias=bias)
    weight, bias = DFlt.tensorize_fn.tensorize(dummy_layer)
    out_linear = (
        dummy_layer(DFlt.attn_input)
        .reshape(DFlt.batch_size, DFlt.seq_len, DFlt.num_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    out_tensor = CPDeApprox.tensor_forward((weight, bias), DFlt.attn_input)
    assert th.allclose(out_tensor, out_linear)


@pytest.mark.parametrize("bias", [True, False])
def test_merged_einsum(bias: bool):
    """Test to check einsum in merged QKV implementation.

    Args:
        bias (bool): Boolean to use bias.
    """
    head_dim: int = DFlt.embed_dim // DFlt.num_heads
    dummy_layer = th.nn.Linear(DFlt.embed_dim, 3 * DFlt.embed_dim, bias=bias)
    weight, bias = DFlt.tensorize_fn.merged_tensorize(dummy_layer)
    out_linear = (
        dummy_layer(DFlt.attn_input)
        .reshape(DFlt.batch_size, DFlt.seq_len, 3, DFlt.num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    out_tensor = CPDeApprox.tensor_forward((weight, bias), DFlt.attn_input, merged=True)
    assert th.allclose(out_tensor, out_linear)


@pytest.mark.parametrize("train_factors", list(range(2, 20)))
def test_split_factors(train_factors: int):
    """Test to check the tensor split.

    Args:
        train_factors (int): Number of trainable rank-1 vectors.
    """
    lambdas, factors = DFlt.cpd_object.decompose(DFlt.cpd_input)
    # Check if lambdas are sorted
    lambda_list = lambdas.tolist()
    assert lambda_list == sorted(lambda_list, reverse=True)
    pretrain, finetune = DFlt.cpd_object.split_factors(
        lambdas=lambdas, factors=factors, trainable_factors=train_factors
    )
    full_tensor = pretrain + finetune
    assert th.allclose(full_tensor, DFlt.cpd_input)


def test_noncp_state_load():
    """Test to check model loading wihtout CP conversion."""
    model = Dummy(cpda=False)
    # Save the existing layer weights
    _create_delete_folder(model, model.save_path, create=True)

    model.apply(_init_ones)

    assert th.allclose(model.final.weight.data, th.ones(model.final.weight.shape))

    state_dict = th.load(f"{model.save_path}/dummy.pt")
    model = DFlt.cpd_object.convert_state_dict(model, state_dict)
    _create_delete_folder(model, model.save_path, create=False)
    assert not th.allclose(model.final.weight.data, th.ones(model.final.weight.shape))
    assert th.allclose(model.final.weight.data, state_dict["final.weight"].data)


def test_cp_state_convert():
    """Test CP conversion during model loading."""
    model = Dummy(cpda=False)
    _create_delete_folder(model, model.save_path, create=True)

    model = Dummy(cpda=True)
    old_state_dict = th.load(f"{model.save_path}/dummy.pt")
    test_weight = old_state_dict["final.weight"]
    DFlt.cpd_object.rank = 10
    DFlt.cpd_object.decomposition = tl.decomposition.CP(
        rank=10, normalize_factors=True, verbose=False, init="random", random_state=10
    )
    tensor_weight = DFlt.cpd_object.tensorize.parameterize(test_weight)
    lambdas, factors = DFlt.cpd_object.decompose(tensor_weight)
    pt, ft = DFlt.cpd_object.split_factors(lambdas, factors, 2)

    model = DFlt.cpd_object.convert_state_dict(model, old_state_dict)

    _create_delete_folder(model, model.save_path, create=False)
    assert th.allclose(model.final.weight.data, pt)
    assert th.allclose(model.final.ft_weight.data, ft)


def test_enable_gradient():
    """Test to enable gradient."""
    model = Dummy(cpda=True)
    DFlt.cpd_object.enable_grad(model)
    assert not model.final.weight.requires_grad
    assert model.final.ft_weight.requires_grad
