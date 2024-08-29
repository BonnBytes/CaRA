"""Tests for utilities."""

import pytest
import torch as th

from src.cpda.cpda import CPDeApprox
from tests.defaults import Defaults as DFlt

th.set_default_dtype(th.float64)


@pytest.mark.parametrize("bias", [False, True])
def test_single_proj_to_tensor(bias: bool):
    """Test to convert single linear projection to tensor.

    Args:
        bias (bool): Boolean to use bias.
    """
    head_dim: int = DFlt.embed_dim // DFlt.num_heads
    q_linear = th.nn.Linear(DFlt.embed_dim, DFlt.embed_dim, bias=bias)
    Wq, Bv = DFlt.tensorize_fn.tensorize(q_linear)

    assert Wq.shape == (DFlt.embed_dim, DFlt.num_heads, head_dim)
    assert Bv.shape == (DFlt.num_heads, head_dim)

    # Typical forward pass
    # Reshape and permtue to (BS, NH, SL, HD)
    Qo = (
        q_linear(DFlt.attn_input)
        .reshape(DFlt.batch_size, DFlt.seq_len, DFlt.num_heads, head_dim)
        .permute(0, 2, 1, 3)
    )

    # FLAX based forward pass
    Qf = CPDeApprox.tensor_forward((Wq, Bv), DFlt.attn_input)

    assert th.allclose(Qf, Qo)


@pytest.mark.parametrize("bias", [False, True])
def test_merged_proj_to_tensor(bias):
    """Test to convert merged linear projection to tensor.

    Args:
        bias (bool): Boolean to use bias.
    """
    head_dim: int = DFlt.embed_dim // DFlt.num_heads
    qkv_linear = th.nn.Linear(DFlt.embed_dim, DFlt.embed_dim * 3, bias=bias)
    (Wq, Wk, Wv), (Bq, Bk, Bv) = DFlt.tensorize_fn.merged_tensorize(qkv_linear)

    assert (
        Wq.shape == Wk.shape == Wv.shape == (DFlt.embed_dim, DFlt.num_heads, head_dim)
    )
    assert Bq.shape == Bk.shape == Bv.shape == (DFlt.num_heads, head_dim)

    # Typical forward pass
    qkv_original = (
        qkv_linear(DFlt.attn_input)
        .reshape(DFlt.batch_size, DFlt.seq_len, 3, DFlt.num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    Qo, Ko, Vo = qkv_original.unbind(0)

    # FLAX based forward pass
    Qf = CPDeApprox.tensor_forward((Wq, Bq), DFlt.attn_input)
    Kf = CPDeApprox.tensor_forward((Wk, Bk), DFlt.attn_input)
    Vf = CPDeApprox.tensor_forward((Wv, Bv), DFlt.attn_input)

    assert th.allclose(Qf, Qo)
    assert th.allclose(Kf, Ko)
    assert th.allclose(Vf, Vo)
