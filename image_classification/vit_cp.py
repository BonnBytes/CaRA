import torch as th
from tqdm import tqdm 
import numpy  as np
from timm.models import create_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from avalanche.evaluation.metrics.accuracy import Accuracy
from vtab import *
from vtab_config import config
import random
import os
import wandb
from timm.scheduler import CosineLRScheduler

from src.cara.cara import cara

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
    acc = Accuracy()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)
    return acc.result()


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
                 "smallnorb_ele", "sun397", "svhn", "dsprites_ori"],
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
    cara_config = {
        "model": vit,
        "rank": args.dim,
        "scale": scale,
        "l_mu": lambda_mean,
        "l_std": lambda_std
        }
    vit = cara(cara_config)
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
    acc = test(vit, test_dl)[1]
    print(acc)
    if acc > args.best_acc:
        args.best_acc = acc
        os.remove(old_name)
        th.save(vit.state_dict(), f"./vit_{name}_{round(args.best_acc, 5)}_seed_{seed}.pt")
    
    print(f"Accuracy: {args.best_acc}")

if __name__ == "__main__":
    main()
