import torch as th
import numpy as np
import random

from src.cara.cara import cara
from timm.models import create_model


def _get_vit():
    return create_model(
        "vit_base_patch16_224_in21k",
        drop_path_rate=0.1
        )


def _get_cara_config():
    random.seed(0)
    th.manual_seed(0)
    np.random.seed(0)
    th.cuda.manual_seed_all(0)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    return {
        "model": _get_vit(),
        "rank": 32,
        "scale": 1.0,
        "l_mu": 1.0,
        "l_std": 0.0
        }


def test_vit_without_cara():
    vit = _get_vit()
    assert (not hasattr(vit, "CP_A1")) and (not hasattr(vit, "CP_A2")) and (not hasattr(vit, "CP_A3")) and (not hasattr(vit, "CP_A4"))
    assert (not hasattr(vit, "CP_P1")) and (not hasattr(vit, "CP_P2")) and (not hasattr(vit, "CP_P3"))
    assert (not hasattr(vit, "CP_R1"))
    assert (not hasattr(vit, "CP_R2"))


def test_vit_with_cara():
    vit = cara(_get_cara_config())
    assert (hasattr(vit, "CP_A1")) and (hasattr(vit, "CP_A2")) and (hasattr(vit, "CP_A3")) and (hasattr(vit, "CP_A4"))
    assert (hasattr(vit, "CP_P1")) and (hasattr(vit, "CP_P2")) and (hasattr(vit, "CP_P3"))
    assert (hasattr(vit, "CP_R1"))
    assert (hasattr(vit, "CP_R2"))


def test_cara_zero_init():
    vit = cara(_get_cara_config())
    assert th.allclose(vit.CP_A2, th.zeros_like(vit.CP_A2))
    assert th.allclose(vit.CP_P2, th.zeros_like(vit.CP_P2))


def test_cara_lambda_init():
    vit = cara(_get_cara_config())
    assert th.allclose(vit.CP_R1, th.ones_like(vit.CP_R1))
    assert th.allclose(vit.CP_R2, th.ones_like(vit.CP_R2))
