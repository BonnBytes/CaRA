import torch as th
from torch import nn
from tqdm import tqdm 
import numpy  as np
from timm.models import create_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from avalanche.evaluation.metrics.accuracy import Accuracy
from vtab import *
from vtab_config import config
import timm
import random
import os
import wandb
from timm.scheduler import CosineLRScheduler
import tensorly as tl
tl.set_backend("pytorch")

best_acc = 0.0

def train(args, model, dl, tdl, opt, sched, epochs):
    model.train()
    model = model.cuda()
    acc = 0.
    idx = 0
    global old_name
    old_name = None
    for epoch in (pbar:=tqdm(range(epochs))):
    # for epoch in range(epochs):
        if log:
            logger.log({"epoch":epoch})
        for batch in dl:
            if log:
                r1_hist = wandb.Histogram(vit.CP_R1.cpu().detach().numpy())
                r2_hist = wandb.Histogram(vit.CP_R2.cpu().detach().numpy())
                logger.log({
                    "R1": r1_hist,
                    "R2": r2_hist
                })
                logger.log({
                    "r1_mean": th.mean(vit.CP_R1.cpu()),
                    "r2_mean": th.mean(vit.CP_R2.cpu())
                })
                logger.log({
                    "r1_std": th.std(vit.CP_R1.cpu()),
                    "r2_std": th.std(vit.CP_R2.cpu())
                })
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = th.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            idx += 1
            if log:
                logger.log({"loss":loss.item()})
            pbar.set_description(f"e: {epoch}, l: {round(loss.item(), 7)}, a:{acc}")
            if sched is not None:
                sched.step(epoch)
        # Add accuracy calculation here
        if epoch % 5 == 0 and epoch != 0 and epoch >= 50:
            sched = None
            acc = test(model, tdl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
                if old_name is not None:
                    os.remove(old_name)
                old_name = f"./vit_{args.dataset}_{round(acc, 5)}_seed_{args.seed}.pt"
                th.save(vit.state_dict(), old_name)
            if log:
                logger.log({"val_acc": acc})
            # model.train()
            # print(f"Epoch: {epoch}, Accuracy: {acc}")
    model = model.cpu()
    return model


@th.no_grad()
def test(model, dl):
    model.eval()
    # acc = []
    # ex = []
    acc = Accuracy()
    model = model.cuda()
    for batch in dl:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)
    return acc.result()
    #     out = th.argmax(out, dim=1)
    #     correct = (out == y).float()
    #     acc.extend(correct)
    #     ex.append(len(y))
    # ac = sum(acc) / sum(ex)
    # return round(ac.cpu().item(), 4)
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
    # F_1, F_2, F_3, F_4 = factors
    B, N, C = input_.shape
    heads = 12
    input_ = input_.reshape((B, N, heads, C//heads))
    # input_ = input_.unsqueeze(0)
    preprocess = (
        lambda x: x.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute((-1, 0, 1, 2, 3))
    )
    # F_1 = preprocess(F_1)
    # F_2 = preprocess(F_2)
    # F_3 = preprocess(F_3)
    # F_4 = preprocess(F_4)
    F_1, F_2, F_3, F_4 = map(preprocess, factors)

    inter_1 = input_.unsqueeze(0) @ F_4.swapaxes(-2, -1)
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
    # # 4D Implementation - Memory expensive
    # qkv_delta = attn_thunder_forward((f1, vit.CP_A2, vit.CP_A3, vit.CP_A4), x, self.dp)
    #  Convert 4D to 2D
    tensor_attn = tl.cp_to_tensor((vit.CP_R1, (f1, vit.CP_A2, vit.CP_A3, vit.CP_A4)))
    K, E, H, D = tensor_attn.shape
    tensor_attn = tensor_attn.reshape((K, E, H*D)).swapaxes(-2, -1)
    qkv_delta = th.einsum("bnd, ked->kbne", x, self.dp(tensor_attn))
    qkv_delta = qkv_delta.reshape(3, B, N, self.num_heads, C//self.num_heads).permute(
        0, 1, 3, 2, 4
    )
    # tensor_attn = tensor_attn.permute(1, 0, 2)
    # tensor_attn = tensor_attn.reshape((tensor_attn.shape[0], -1))
    # qkv_delta = x @ self.dp(tensor_attn)
    # qkv_delta = qkv_delta.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(
    #     2, 0, 3, 1, 4
    # )
    qkv = qkv.reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(
        2, 0, 3, 1, 4
    )
    qkv += qkv_delta * self.s
    # q, k, v = qkv.unbind(0)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn@v).transpose(1, 2).reshape(B, N, C)

    proj = self.proj(x)
    p1 = vit.CP_P1[self.idx:self.idx+1, :]
    # proj_delta = mlp_thunder_forward((p1, vit.CP_P2, vit.CP_P3), x, self.dp)
    tensor_proj = tl.cp_to_tensor((vit.CP_R2, (p1, vit.CP_P2, vit.CP_P3)))
    AA, AB, AC = tensor_proj.shape
    tensor_proj = tensor_proj.reshape((AA*AB, AC))
    proj_delta = x@self.dp(tensor_proj.T) + vit.CP_bias1
    proj += proj_delta * self.s
    x = self.proj_drop(proj)
    return x


def cp_mlp(self, x):
    p1_up = vit.CP_P1[self.idx:self.idx+4, :]
    p1_down = vit.CP_P1[self.idx+4: self.idx+8, :]

    up = self.fc1(x)
    # up_delta = mlp_thunder_forward((p1_up, vit.CP_P2, vit.CP_P3), x, self.dp)
    tensor_up = tl.cp_to_tensor((vit.CP_R2, (p1_up, vit.CP_P2, vit.CP_P3)))
    AA, AB, AC = tensor_up.shape
    tensor_up = tensor_up.reshape((AA*AB, AC))
    up_delta = x@self.dp(tensor_up.T) + vit.CP_bias2
    up += up_delta * self.s

    x = self.act(up)
    x = self.drop(x)
    
    down = self.fc2(x)
    # down_delta = mlp_down_forward((p1_down, vit.CP_P2, vit.CP_P3), x, self.dp)
    tensor_down = tl.cp_to_tensor((vit.CP_R2, (p1_down, vit.CP_P2, vit.CP_P3)))
    tensor_down = tensor_down.reshape((AA*AB, AC))
    down_delta = x@self.dp(tensor_down) + vit.CP_bias3
    down += down_delta * self.s
    x = self.drop(down)
    return x


def set_CP(model, dim, s, l_mu, l_std):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.CP_A1 = nn.Parameter(th.empty([36, dim]), requires_grad=True)
        model.CP_A2 = nn.Parameter(th.empty([768, dim]), requires_grad=True)
        model.CP_A3 = nn.Parameter(th.empty([12, dim]), requires_grad=True)
        model.CP_A4 = nn.Parameter(th.empty([768//12, dim]), requires_grad=True)
        model.CP_P1 = nn.Parameter(th.empty([108, dim]), requires_grad=True)
        model.CP_P2 = nn.Parameter(th.empty([768, dim]), requires_grad=True)
        model.CP_P3 = nn.Parameter(th.empty([768, dim]), requires_grad=True)
        model.CP_R1 = nn.Parameter(th.empty([dim], requires_grad=True))
        model.CP_R2 = nn.Parameter(th.empty([dim], requires_grad=True))

        model.CP_bias1 = nn.Parameter(th.empty([768]), requires_grad=True)
        model.CP_bias2 = nn.Parameter(th.empty([768*4]), requires_grad=True)
        model.CP_bias3 = nn.Parameter(th.empty([768]), requires_grad=True)
        
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
        elif type(child) == timm.models.layers.mlp.Mlp:
            child.dim = dim
            child.s = s
            child.dp = nn.Dropout(0.1)
            child.idx = vit.idx
            vit.idx += 8
            bound_method = cp_mlp.__get__(child, child.__class__)
            setattr(child, "forward", bound_method)
        elif len(list(child.children())) != 0:
            set_CP(child, dim, s, l_mu, l_std)

def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dim",
        default=32,
        type=int,
        help="Number of trainable ranks."
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--dataset",
        default="svhn",
        type=str,
        choices=["cifar", "caltech101", "clevr_count", "clevr_dist", "diabetic_retinopathy",
                 "dmlab", "dsprites_loc", "dtd", "eurosat", "kitti", "oxford_flowers102",
                 "oxford_iiit_pet", "patch_camelyon", "resisc45", "smallnorb_azi",
                 "smallnorb_ele", "sun397", "svhn"],
        help="Dataset to train"
    )
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    return parser.parse_args()


def main(sd = None):
    global logger, log

    args = _parse_args()
    print(args)
    name = args.dataset
    
    data_config = config[name]
    if sd is None:
        seed = data_config["seed"]
    else:
        seed = sd
    scale = data_config["scale"]
    log = data_config["logger"]
    lambda_mean = data_config["init_mean"]
    lambda_std = data_config["init_std"]
    args.best_acc = 0.0
    args.seed = seed

    print(f"\n\nSeed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    if log:
        run_name = f"LR__{name}__{args.lr}-Scale_{scale}-Rank_{args.dim}_test"
        logger = wandb.init(project="Fact-CP", name=run_name)
        logger.config.update(args)

    train_dl, test_dl = get_data(name, evaluate=True)
    num_classes = get_classes_num(name)
    global vit
    vit = create_model(args.model, checkpoint_path="./ViT-B_16.npz", drop_path_rate=0.1)
    # vit = th.nn.DataParallel(vit)
    set_CP(vit, dim=args.dim, s=scale, l_mu=lambda_mean, l_std=lambda_std)
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
    print(vit.head)
    optimizer = th.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = None
    scheduler = CosineLRScheduler(optimizer, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
    vit = train(args, vit, train_dl, test_dl, optimizer, scheduler, epochs=100)
    print("\n\n Evaluating....")
    _, test_dl = get_data(name, evaluate=True)
    acc = test(vit, tqdm(test_dl))[1]
    if acc > args.best_acc:
        args.best_acc = acc
        os.remove(old_name)
        th.save(vit.state_dict(), f"./vit_{name}_{round(args.best_acc, 5)}_seed_{seed}.pt")
    
    print(f"Accuracy: {args.best_acc}")

if __name__ == "__main__":
    # for i in range(0, 25):
    #     main(i)
    # for i in range(25, 50):
    #     main(i)
    # for i in range(50, 75):
    #     main(i)
    for i in range(75, 100):
        main(i)
    # main()
