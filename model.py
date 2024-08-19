import torch as th

import tensorly as tl
tl.set_backend("pytorch")
from typing import Tuple
# RANK = 500


# class Dummy(th.nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super().__init__()
#         # self.lin1 = th.nn.Linear(3675, 4096)
#         self.lin1 = CPLinear(3675, 4096, r=RANK)
#         # self.lin2 = th.nn.Linear(4096, 2048)
#         self.lin2 = CPLinear(4096, 2048, r=RANK)
#         # self.lin3 = th.nn.Linear(2048, 1024)
#         self.lin3 = CPLinear(2048, 1024, r=RANK)
#         self.lin4 = th.nn.Linear(1024, num_classes)
#         self.relu = th.nn.ReLU()
#         self.sigm = th.nn.Sigmoid()
    
#     def forward(self, x):
#         x = x.reshape((x.shape[0], -1))
#         x = self.relu(self.lin1(x))
#         x = self.sigm(self.lin2(x))
#         x = self.relu(self.lin3(x))
#         return self.lin4(x)


class CPLoRA(th.nn.Module):
    def __init__(self, tr_rank: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        self.tr_rank = tr_rank
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.S_model_ft = th.nn.Parameter(th.zeros((embed_dim, tr_rank)), requires_grad=True)
        self.S_heads_ft = th.nn.Parameter(th.zeros((num_heads, tr_rank)), requires_grad=True)
        self.S_headdim_ft = th.nn.Parameter(th.zeros((head_dim, tr_rank)), requires_grad=True)
        # self.lambdas_ft = th.nn.Parameter(th.ones((tr_rank)), requires_grad=True)
        self.relu = th.nn.ReLU()
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape((B, N, self.num_heads, self.head_dim))
        # op2 = self.__thunder_forward((self.lambdas_ft, self.S_model_ft, self.S_heads_ft, self.S_headdim_ft), x)
        op2 = self.__thunder_forward((self.S_model_ft, self.S_heads_ft, self.S_headdim_ft), x)
        return op2
        # tensor_ = tl.cp_to_tensor((self.lambdas_ft, (self.S_model_ft, self.S_heads_ft, self.S_headdim_ft)))
        # op2 = self._tensor_forward(tensor_, x)
        # return op2.reshape((B, N, C))

    def _tensor_forward(self, tensor, input_):
        return th.einsum("bne, ehd -> bnhd", (input_, tensor))

    def __thunder_forward(
        self, factors: Tuple[th.nn.Parameter, ...], input_: th.Tensor
    ) -> th.Tensor:
        """Compute projection using CP Factors.

        Args:
            factors (Tuple[th.nn.Parameter, ...]): Tuple containing factors.
            input_ (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Projected output.
        """
        # assert len(factors) == 4
        # lambdas, S_e, S_h, S_d = factors
        S_e, S_h, S_d = factors
        input_ = input_.unsqueeze(0)  # (1, bs, patches, heads, headdim)
        preprocess = (
            lambda x: x.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute([-1, 0, 1, 2, 3])
        )
        S_d = preprocess(S_d)
        S_h = preprocess(S_h).squeeze(-2)
        S_e = preprocess(S_e).squeeze(-2)
        # CP Form forward pass
        inter_1 = input_ @ S_d.swapaxes(-2, -1)  # (rank, bs, patches, heads, 1)
        inter_1 = inter_1.squeeze(-1)  # (rank, bs, patches, heads)
        inter_2 = inter_1 @ S_h.swapaxes(-2, -1)  # (rank, bs, patches, 1)
        # output_ = lambdas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (
        #     inter_2 @ S_e
        # )  # (rank, bs, patches, d_model)
        output_ = inter_2 @ S_e  # (rank, bs, patches, d_model)
        output_ = th.sum(output_, 0)  # (bs, patches, d_model)
        return output_


class CPLoraMerged(th.nn.Linear):
    def __init__(self, in_features: int, out_features: int, tr_rank: int, embed_dim: int, num_heads: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert out_features == 3*in_features
        self.tr_rank = tr_rank
        self.CPQ = CPLoRA(tr_rank, embed_dim, num_heads)
        self.CPK = CPLoRA(tr_rank, embed_dim, num_heads)
        self.CPV = CPLoRA(tr_rank, embed_dim, num_heads)
    
    def forward(self, x):
        out1 = super().forward(x)
        Q = self.CPQ(x)
        K = self.CPK(x)
        V = self.CPV(x)
        out = th.cat([Q, K, V], dim=-1)
        return out1 + out
