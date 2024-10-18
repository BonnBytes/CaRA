import torch as th
from torch import nn
from tqdm import tqdm 
import numpy  as np
from timm.models import create_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from vtab import *
import timm
from timm.scheduler import CosineLRScheduler
import tensorly as tl
tl.set_backend("pytorch")


def train(args, model, dl, tdl, opt, sched, epochs):
    model.train()
    model = model.cuda()
    acc = 0.
    for epoch in (pbar:=tqdm(range(epochs))):
        for batch in dl:
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = th.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(f"e: {epoch}, l: {round(loss.item(), 4)}, a:{acc}")
        if sched is not None:
            sched.step(epoch)
        # Add accuracy calculation here
        if epoch % 25 == 0 and epoch != 0:
            acc = test(model, tdl)
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
        acc.extend(correct)
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


def attn_thunder_forward(factors, input_, dropout=None):
    if dropout is None:
        dropout = nn.Identity()
    F_1, F_2, F_3, F_4 = factors
    B, N, C = input_.shape
    heads = 12
    input_ = input_.reshape((B, N, heads, C//heads))
    input_ = input_.unsqueeze(0)
    preprocess = (
        lambda x: x.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute((-1, 0, 1, 2, 3))
    )
    F_1 = preprocess(F_1)
    F_2 = preprocess(F_2)
    F_3 = preprocess(F_3)
    F_4 = preprocess(F_4)

    inter_1 = input_ @ F_4.swapaxes(-2, -1)
    del F_4
    inter_2 = F_3 @ inter_1
    del F_3, inter_1
    inter_3 = inter_2 @ F_2
    del F_2, inter_2
    output = F_1.swapaxes(-2, -1) @ dropout(inter_3)
    del F_1, inter_3
    output = th.sum(output, 0).permute((2, 0, 1, 3))
    K, B, N, C = output.shape
    return output.reshape((K, B, N, heads, C//heads)).permute(0, 1, 3, 2, 4)


def mlp_thunder_forward(factors, input_, dropout = None):
    if dropout is None:
        dropout = nn.Identity()
    P_1, P_2, P_3 = factors
    B, N, C = input_.shape
    input_ = input_.unsqueeze(0).unsqueeze(-2)
    preprocess = (
        lambda x: x.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute((-1, 0, 1, 2, 3))
    )
    P_1 = preprocess(P_1)
    P_2 = preprocess(P_2)
    P_3 = preprocess(P_3)

    inter_1 = input_ @ P_3.swapaxes(-2, -1)
    del P_3
    inter_2 = inter_1 @ dropout(P_2)
    del P_2, inter_1
    output = P_1.swapaxes(-2, -1) @ inter_2
    del P_1, inter_2
    R, B, N, e, k = output.shape
    output = output.reshape((R, B, N, e*k))
    output = th.sum(output, dim=0)
    return output


def mlp_down_forward(factors, input_, dropout=None):
    if dropout is None:
        dropout = nn.Identity()
    P_1, P_2, P_3 = factors
    B, N, C = input_.shape
    x_ = input_.reshape(B, N, C//4, 4)
    x_ = x_.unsqueeze(0)
    preprocess = (
        lambda x: x.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute((-1, 0, 1, 2, 3))
    )
    P_1 = preprocess(P_1)
    P_2 = preprocess(P_2)
    P_3 = preprocess(P_3)
    inter_1 = x_ @ P_1.swapaxes(-2, -1)
    del P_1
    inter_2 = dropout(P_2) @ inter_1
    del P_2, inter_1
    inter_2 = inter_2.squeeze(-1)
    output = inter_2 @ P_3.squeeze(1)
    del P_3, inter_2
    output = th.sum(output, dim=0)
    return output


def cp_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)
    f1 = vit.CP_A1[self.attn_idx:self.attn_idx+3, :]
    qkv_delta = attn_thunder_forward((f1, vit.CP_A2, vit.CP_A3, vit.CP_A4), x, self.dp)

    qkv = qkv.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(
        2, 0, 3, 1, 4
    )
    qkv += qkv_delta * self.s
    q, k, v = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn@v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    p1 = vit.CP_P1[self.idx:self.idx+1, :]
    # proj_delta = mlp_thunder_forward((p1, vit.CP_P2, vit.CP_P3), x, self.dp)
    tensor_proj = tl.cp_to_tensor((None, (p1, vit.CP_P2, vit.CP_P3)))
    AA, AB, AC = tensor_proj.shape
    tensor_proj = tensor_proj.reshape((AA*AB, AC))
    proj_delta = x@self.dp(tensor_proj.T)
    proj += proj_delta * self.s
    x = self.proj_drop(proj)
    return x


def cp_mlp(self, x):
    p1_up = vit.CP_P1[self.idx:self.idx+4, :]
    p1_down = vit.CP_P1[self.idx+4: self.idx+8, :]

    up = self.fc1(x)
    # up_delta = mlp_thunder_forward((p1_up, vit.CP_P2, vit.CP_P3), x, self.dp)
    tensor_up = tl.cp_to_tensor((None, (p1_up, vit.CP_P2, vit.CP_P3)))
    AA, AB, AC = tensor_up.shape
    tensor_up = tensor_up.reshape((AA*AB, AC))
    up_delta = x@self.dp(tensor_up.T)
    up += up_delta * self.s

    x = self.act(up)
    x = self.drop1(x)
    
    down = self.fc2(x)
    # down_delta = mlp_down_forward((p1_down, vit.CP_P2, vit.CP_P3), x, self.dp)
    tensor_down = tl.cp_to_tensor((None, (p1_down, vit.CP_P2, vit.CP_P3)))
    tensor_down = tensor_down.reshape((AA*AB, AC))
    down_delta = x@self.dp(tensor_down)
    down += down_delta * self.s
    x = self.drop2(down)
    return x


def set_CP(model, dim=9, s=1):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.CP_A1 = nn.Parameter(th.empty([36, dim]), requires_grad=True)
        model.CP_A2 = nn.Parameter(th.empty([768, dim]), requires_grad=True)
        model.CP_A3 = nn.Parameter(th.empty([12, dim]), requires_grad=True)
        model.CP_A4 = nn.Parameter(th.empty([768//12, dim]), requires_grad=True)
        model.CP_P1 = nn.Parameter(th.empty([108, dim]), requires_grad=True)
        model.CP_P2 = nn.Parameter(th.empty([768, dim]), requires_grad=True)
        model.CP_P3 = nn.Parameter(th.empty([768, dim]), requires_grad=True)
        
        nn.init.xavier_normal_(model.CP_A1)
        nn.init.xavier_normal_(model.CP_A2)
        nn.init.xavier_normal_(model.CP_A3)
        nn.init.xavier_normal_(model.CP_A4)
        nn.init.xavier_normal_(model.CP_P1)
        nn.init.xavier_normal_(model.CP_P2)
        nn.init.xavier_normal_(model.CP_P3)
        model.idx = 0
        model.attn_idx = 0
    for child in model.children():
        if type(child) == timm.models.vision_transformer.Attention:
            child.dp = nn.Dropout(0.1)
            child.s = s
            child.dim = dim
            child.idx = vit.idx
            child.attn_idx = vit.attn_idx
            vit.idx += 1
            vit.attn_idx += 3
            bound_method = cp_attn.__get__(child, child.__class__)
            setattr(child, "forward", bound_method)
        elif type(child) == timm.layers.mlp.Mlp:
            child.dim = dim
            child.s = s
            child.dp = nn.Dropout(0.1)
            child.idx = vit.idx
            vit.idx += 8
            bound_method = cp_mlp.__get__(child, child.__class__)
            setattr(child, "forward", bound_method)
        elif len(list(child.children())) != 0:
            set_CP(child, dim, s)

def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dim",
        default=16,
        type=int,

        help="Number of trainable ranks."
    )
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    return parser.parse_args()


def main():
    th.manual_seed(42)
    args = _parse_args()
    print(args)
    name = "svhn"
    train_dl, test_dl = get_data(name, evaluate=True)
    num_classes = get_classes_num(name)
    global vit
    vit = create_model(args.model, checkpoint_path="./ViT-B_16.npz", drop_path_rate=0.1)
    # vit = th.nn.DataParallel(vit)
    set_CP(vit, dim=args.dim, s=1)
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
    # optimizer = th.optim.SGD(trainable, lr=1e-2, momentum=0.8, nesterov=True)
    scheduler = None
    # scheduler = CosineLRScheduler(optimizer, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, k_decay=0.1)
    vit = train(args, vit, train_dl, test_dl, optimizer, scheduler, epochs=100)
    print("\n\n Evaluating....")
    _, test_dl = get_data(name, evaluate=True)
    acc = test(vit, tqdm(test_dl))
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    main()
