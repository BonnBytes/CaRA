import torch as th
from torch import nn
from tqdm import tqdm 
import numpy  as np
import random
import _testimportmultiple
from timm.models import create_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from vtab import *
import timm


def train(args, model, dl, tdl, opt, sched, epochs):
    model.train()
    model = model.cuda()
    for epoch in (pbar:=tqdm(range(epochs))):
        for _, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = th.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched is not None:#
            sched.step(epoch)
        # Add accuracy calculation here
        if epoch % 50 == 0:
            acc = test(model, tdl)
            print(f"Epoch: {epoch}, Accuracy: {acc}")
            pbar.set_description(f"Accuracy: {acc}")
    model = model.cpu()
    return model


@th.no_grad()
def test(model, dl):
    model.eval()
    acc = []
    ex = []
    model = model.cuda()
    for batch in dl:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        out = th.argmax(out, dim=1)
        correct = (out == y).float()
        acc.append(correct)
        ex.append(len(y))
    ac = sum(acc) / sum(ex)
    return round(ac.cpu().item(), 4)
    # Add accuracy calculation here


def split_weight(weight):
    d1, d2 = weight.shape
    if d1 != 3 * d2:
        raise RuntimeError("Weight out dimension is not 3 times its input dimensions.")
    d1 //= 3
    layer_weight = weight.reshape(3, d1, d2)
    return layer_weight.unbind(0)


def thunder_forward(factors, input_):
    S_e, S_h, S_d = factors
    input_ = input_.unsqueeze(0)
    preprocess = (
        lambda x: x.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute([-1, 0, 1, 2, 3])
    )
    S_d = preprocess(S_d)
    S_h = preprocess(S_h).squeeze(-2)
    S_e = preprocess(S_e).squeeze(-2)

    inter_1 = input_ @ S_d.swapaxes(-2, -1)
    inter_1 = inter_1.squeeze(-1)
    inter_2 = inter_1 @ S_h.swapaxes(-2, -1)
    output_ = inter_2 @ S_e
    output_ = th.sum(output_, 0)
    return output_


def cp_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)
    q = thunder_forward((self.CP_E, self.CP_H, self.CP_D), x)
    k = thunder_forward((self.CP_E, self.CP_H, self.CP_D), x)
    v = thunder_forward((self.CP_E, self.CP_H, self.CP_D), x)
    qkv += torch.cat([q, k, v], dim=2) * self.s

    qkv = qkv.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn@v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    x = self.proj_drop(proj)
    return x


def cp_mlp(self, x):
    pass

def set_CP(model, dim=9, s=1):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.CP_E = nn.Linear(768, dim, bias=False)
        model.CP_H = nn.Linear(12, dim, bias=False)
        model.CP_D = nn.Linear(768//12, dim, bias=False)

        nn.init.xavier_normal_(model.CP_E)
        nn.init.xavier_normal_(model.CP_H)
        nn.init.xavier_normal_(model.CP_D)
        model.idx = 0
        for child in model.children():
            if type(child) == timm.models.vision_transformer.Attention:
                child.dp = nn.Dropout(0.1)
                child.s = s
                child.dim = dim
                child.idx = vit.idx
                vit.idx += 4
                bound_method = cp_attn.__get__(child, child.__class__)
                setattr(child, "forward", bound_method)
            elif len(list(child.children())) != 0:
                set_CP(child, dim, s)

def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dim",
        default=2,
        type=int,

        help="Number of trainable ranks."
    )
    return parser.parse_args()


def main():
    th.manual_seed(42)
    args = _parse_args()
    print(args)
    name = "svhn"
    train_dl, test_dl = get_data(name)
    num_classes = get_classes_num(name)

    vit = create_model("vit_base_patch16_224_in21k", checkpoint_path="./ViT-B_16.npz", drop_path_rate=0.1)
    set_CP(vit, dim=2, s=1)
    trainable = []
    vit.reset_classifier(num_classes)
    total_param = 0
    for n, p in vit.named_parameters():
        if "CP" in n or "head" in n:
            trainable.append(p)
            if "head" not in n:
                total_param += p.numel()
        else:
            p.requires_grad = False

    print(f"Total parameters: {total_param}")
    optimizer = th.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    scheduler = None
    vit = train(args, vit, train_dl, test_dl, optimizer, scheduler, epoch=100)
