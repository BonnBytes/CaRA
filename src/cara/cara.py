from copy import deepcopy
from typing import Dict, Any

import torch as th
import torch.nn as nn
import timm
import tensorly as tl
tl.set_backend("pytorch")


def cp_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)
    f1 = global_model.CP_A1[self.attn_idx:self.attn_idx+3, :]
    tensor_attn = tl.cp_to_tensor((global_model.CP_R1, (f1, global_model.CP_A2, global_model.CP_A3, global_model.CP_A4)))
    K, E, H, D = tensor_attn.shape
    tensor_attn = tensor_attn.reshape((K, E, H*D))
    qkv_delta = th.einsum("bnd, kde->kbne", x, self.dp(tensor_attn))
    qkv_delta = qkv_delta.reshape(3, B, N, self.num_heads, C//self.num_heads).permute(
        0, 1, 3, 2, 4
    )
    qkv = qkv.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(
        2, 0, 3, 1, 4
    )
    qkv += qkv_delta * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn@v).transpose(1, 2).reshape(B, N, C)

    proj = self.proj(x)
    p1 = global_model.CP_P1[self.idx:self.idx+1, :]
    tensor_proj = tl.cp_to_tensor((global_model.CP_R2, (p1, global_model.CP_P2, global_model.CP_P3)))
    AA, AB, AC = tensor_proj.shape
    tensor_proj = tensor_proj.reshape((AA*AB, AC))
    proj_delta = x@self.dp(tensor_proj.T) + global_model.CP_bias1
    proj += proj_delta * self.s
    x = self.proj_drop(proj)
    return x


def cp_mlp(self, x):
    p1_up = global_model.CP_P1[self.idx:self.idx+4, :]
    p1_down = global_model.CP_P1[self.idx+4: self.idx+8, :]

    up = self.fc1(x)
    tensor_up = tl.cp_to_tensor((global_model.CP_R2, (p1_up, global_model.CP_P2, global_model.CP_P3)))
    AA, AB, AC = tensor_up.shape
    tensor_up = tensor_up.reshape((AA*AB, AC))
    up_delta = x@self.dp(tensor_up.T) + global_model.CP_bias2
    up += up_delta * self.s

    x = self.act(up)
    x = self.drop(x)
    
    down = self.fc2(x)
    tensor_down = tl.cp_to_tensor((global_model.CP_R2, (p1_down, global_model.CP_P2, global_model.CP_P3)))
    tensor_down = tensor_down.reshape((AA*AB, AC))
    down_delta = x@self.dp(tensor_down) + global_model.CP_bias3
    down += down_delta * self.s
    x = self.drop(down)
    return x


def set_cara(model: nn.Module, rank: int, scale: float, l_mu: float, l_std: float):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        # Declare CaRA parameters
        model.CP_A1 = nn.Parameter(th.empty([36, rank]), requires_grad=True)
        model.CP_A2 = nn.Parameter(th.empty([768, rank]), requires_grad=True)
        model.CP_A3 = nn.Parameter(th.empty([12, rank]), requires_grad=True)
        model.CP_A4 = nn.Parameter(th.empty([768//12, rank]), requires_grad=True)
        model.CP_P1 = nn.Parameter(th.empty([108, rank]), requires_grad=True)
        model.CP_P2 = nn.Parameter(th.empty([768, rank]), requires_grad=True)
        model.CP_P3 = nn.Parameter(th.empty([768, rank]), requires_grad=True)
        model.CP_R1 = nn.Parameter(th.empty([rank]), requires_grad=True)
        model.CP_R2 = nn.Parameter(th.empty([rank]), requires_grad=True)
        model.CP_bias1 = nn.Parameter(th.empty([768]), requires_grad=True)
        model.CP_bias2 = nn.Parameter(th.empty([768*4]), requires_grad=True)
        model.CP_bias3 = nn.Parameter(th.empty([768]), requires_grad=True)
        # Initialise CaRA parameters
        nn.init.xavier_normal_(model.CP_A1)
        nn.init.zeros_(model.CP_A2)
        nn.init.orthogonal_(model.CP_A3)
        nn.init.orthogonal_(model.CP_A4)
        nn.init.xavier_normal_(model.CP_P1)
        nn.init.zeros_(model.CP_P2)
        nn.init.orthogonal_(model.CP_P3)
        if l_std != 0.0:
            nn.init.normal_(model.CP_R1, mean=l_mu, std=l_std)
            nn.init.normal_(model.CP_R2, mean=l_mu, std=l_std)
        elif l_mu == 1.0 and l_std == 0.0:
            nn.init.ones_(model.CP_R1)
            nn.init.ones_(model.CP_R2)
        nn.init.zeros_(model.CP_bias1)
        nn.init.zeros_(model.CP_bias2)
        nn.init.zeros_(model.CP_bias3)
        # CaRA indexing
        model.idx = 0
        model.attn_idx = 0
    for child in model.children():
        if type(child) == timm.models.vision_transformer.Attention:
            child.dp = nn.Dropout(0.1)
            child.s = scale
            child.dim = rank
            child.idx = global_model.idx
            child.attn_idx = global_model.attn_idx
            global_model.idx += 1
            global_model.attn_idx += 3
            bound_method = cp_attn.__get__(child, child.__class__)
            setattr(child, "forward", bound_method)
        elif type(child) == timm.models.layers.mlp.Mlp:
            child.dp = nn.Dropout(0.1)
            child.s = scale
            child.dim = rank
            child.idx = global_model.idx
            global_model.idx += 8
            bound_method = cp_mlp.__get__(child, child.__class__)
            setattr(child, "forward", bound_method)
        elif len(list(child.children())) != 0:
            set_cara(child, rank, scale, l_mu, l_std)
            

def cara(config):
    # CaRA parameters
    model = config["model"]
    rank = config["rank"]
    scale = config["scale"]
    l_mu = config["l_mu"]
    l_std = config["l_std"]
    
    global global_model
    global_model = model
    set_cara(model, rank, scale, l_mu, l_std)
    return global_model