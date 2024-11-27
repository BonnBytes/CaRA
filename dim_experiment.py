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
from timm.models.vision_transformer import Attention
import tensorly as tl
import torch.nn.functional as F
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


def mod_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    x_out = x@self.qkv_tuned.T + self.qkv.bias
    qkv = x_out.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class VitWrapper(nn.Module):
    def __init__(self, vit: nn.Module, cp_summand_length: int = 5):
        # loop through attention heads.
        super().__init__()
        self.vit = vit
        parameter_list = []
        self.cp_summand_length = cp_summand_length

        for mod in self.vit.modules():
            if type(mod) is Attention:
               qkv_un_split = torch.stack(torch.split(mod.qkv.weight, mod.qkv.weight.shape[0]//3, dim=0), dim=0)
               parameter_list.append(torch.stack(torch.split(qkv_un_split, qkv_un_split.shape[1]//mod.num_heads, -2),
                                                 dim=0))
        parameter_tensor = torch.stack(parameter_list).cpu()
        self.stack_shape = parameter_tensor.shape

        print(self.stack_shape)
        if cp_summand_length < len(self.stack_shape):
            parameter_tensor_re = torch.reshape(parameter_tensor,
                                                [-1] + list(self.stack_shape[-cp_summand_length:]) )
            print(f'reshaped-coda-block: {parameter_tensor_re.shape}')
        else:
            parameter_tensor_re = parameter_tensor

        param_prod_fun = lambda t: np.prod(t.shape)

        cp_fun = tl.decomposition.CP(args.ranks, init='random', normalize_factors=False)
        factors = cp_fun.fit_transform(parameter_tensor_re)
        params = sum(factors.weights.shape) + sum(map(param_prod_fun, factors.factors))
        print(f"Parameter count: {params}.")

        for fact in factors.factors:
            nn.init.xavier_normal_(fact)

        factors.factors[-1] = factors.factors[-1]*0. 
        self.weights = th.nn.Parameter(torch.ones_like(factors.weights), requires_grad=True)
        self.factors = torch.nn.ParameterList([th.nn.Parameter(f_vecs, requires_grad=True)
                                               for f_vecs in factors.factors])
        # breakpoint()
        pass

    def restore_parameter_tensor(self):
        undo_cp = tl.cp_to_tensor((self.weights, self.factors))
        undo_cp = th.reshape(undo_cp, self.stack_shape)
        return undo_cp

    def forward(self, x):
        undo_cp = self.restore_parameter_tensor()

        layer = 0
        for mod in self.vit.modules():
            if type(mod) is Attention:
                tune_val = undo_cp[layer]
                mod.qkv_tuned = mod.qkv.weight + scale*tune_val.reshape(mod.qkv.weight.shape)
                layer += 1
                # mod.forward = lambda x: mod_forward(mod, x)
                bound_method = mod_forward.__get__(mod, mod.__class__)
                setattr(mod, "forward", bound_method)

        return self.vit(x)




def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ranks",
        default=32,
        type=int,
        help="Number of trainable CP-ranks."
    )
    parser.add_argument(
        "--dims",
        default=5,
        type=int,
        help="The number of CP-Factors"
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

    global args
    args = _parse_args()
    print(args)
    name = args.dataset
    
    data_config = config[name]
    if sd is None:
        seed = data_config["seed"]
    else:
        seed = sd
    global scale 
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
    # set_CP(vit, dim=args.dim, s=scale, l_mu=lambda_mean, l_std=lambda_std)
    vit.reset_classifier(num_classes)

    for p in vit.parameters():
        p.requires_grad = False
    

    print(f"vit_head: {vit.head}.")
    
    vit = VitWrapper(vit, args.dims)
    vit.cuda()

    # for n, p in vit.named_parameters():
    #     if 'bias' in n:
    #         p.requires_grad = True

    optimizer = th.optim.AdamW(vit.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    scheduler = CosineLRScheduler(optimizer, t_initial=100, warmup_t=10,
                                  lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
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
    main()
