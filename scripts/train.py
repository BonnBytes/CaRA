"""CIFAR10 training script."""

import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import loralib as lora
import torch as th
import torch.nn as nn
import torchvision.datasets as tvds
import torchvision.transforms as tvtf
from torch.utils.tensorboard.writer import SummaryWriter

from dataloader import get_data
from models.classification.vision_transformer import vit_base_patch16_224

parser = argparse.ArgumentParser(description="Train CIFAR100-Vision Transformer.")
parser.add_argument(
    "--epochs", default=100, type=int, help="Total number of epochs to run."
)
parser.add_argument(
    "-b", "--batch-size", default=64, type=int, help="Mini batch size (default: 256)."
)
parser.add_argument(
    "--lr", "--learning-rate", default=1e-3, type=float, help="Learning rate."
)
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD.")
parser.add_argument(
    "--resume", action="store_true", default=False, help="Resume training."
)
parser.add_argument(
    "--tensorboard",
    action="store_true",
    default=True,
    help="Log progress to Tensorboard.",
)
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Run only on CPU."
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="Random seed (default: 42)."
)
parser.add_argument(
    "--model", default="VIST", choices=["VIST"], type=str, help="Model to be optimized."
)
parser.add_argument(
    "--peft",
    default="LORA",
    choices=["FT", "LORA", "CPDA"],
    type=str,
    help="PEFT model to use.",
)
parser.add_argument(
    "--print-every", default=50, type=int, help="Print frequency (default: 100)."
)
parser.add_argument(
    "--clip-norm", default=1, type=int, help="Norm value for gradient clipping."
)

# New argument for dataset name
parser.add_argument(
    "--dataset",
    default="SVHN",
    choices=["SVHN", "CIFAR100", "CIFAR10"],
    type=str,
    help="Dataset to use.",
)

ARGS = parser.parse_args()
DEVICE = (
    th.device("cuda") if th.cuda.is_available() and not ARGS.cpu else th.device("cpu")
)
print(ARGS, flush=True)

if ARGS.tensorboard:
    time = str(datetime.now())
    WRITER = SummaryWriter(
        comment="_"
        + "_lr_"
        + str(ARGS.lr)
        + "_seed_"
        + str(ARGS.seed)
        + "_peft_"
        + ARGS.peft
        + "_time_"
        + time
        + "_model_"
        + ARGS.model
        + "_dataset_"
        + ARGS.dataset
    )


def main():
    """CIFAR10 main."""
    global ARGS, DEVICE
    th.manual_seed(ARGS.seed)
    # Data loading
    normalize = tvtf.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    train_tf = tvtf.Compose(
        [
            tvtf.Resize((224, 224)),
            # tvtf.RandomHorizontalFlip(),
            tvtf.ToTensor(),
            normalize,
        ]
    )
    test_tf = tvtf.Compose(
        [
            tvtf.Resize((224, 224)),
            tvtf.ToTensor(),
            normalize,
        ]
    )

    kwargs = {"num_workers": 1, "pin_memory": True}

    num_classes = 0
    if ARGS.dataset == "CIFAR100":
        num_classes = 100
    else:
        num_classes = 10

    train_loader, val_loader = get_data(ARGS.dataset)

    model = vit_base_patch16_224(pretrained=True)
    # Change the classification head.
    model.head = nn.Linear(model.head.in_features, num_classes)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    for param in model.head.parameters():
        param.requires_grad = True
    lora.mark_only_lora_as_trainable(model)
    for param in model.head.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    total_params = round(total_params / 1000000, 4)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = round(trainable_params / 1000000, 4)
    print(f"Total #Parameters: {total_params}M")
    print(f"Trainable #Parameters: {trainable_params}M")

    loss_fn = th.nn.CrossEntropyLoss()
    # opt = th.optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    opt = th.optim.AdamW(model.parameters(), lr=ARGS.lr)

    model = model.to(DEVICE)

    for epoch in range(ARGS.epochs):
        print(f"Epoch: {epoch}")
        model.train()
        train(train_loader, model, loss_fn, opt, epoch)
        model.eval()
        validate(val_loader, model, loss_fn, epoch)


def train(
    train_loader: th.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: th.optim.Optimizer,
    epoch: int,
):
    """Model training step.

    Args:
        train_loader (th.utils.data.DataLoader): Training dataloader.
        model (nn.Module): Model objekt.
        criterion (nn.Module): Loss objekt.
        optimizer (th.optim.Optimizer): Optimizer objekt.
        epoch (int): Current epoch.
    """
    closses = AverageMeter()
    top1 = AverageMeter()

    for i, (ip, target) in enumerate(train_loader):
        ip, target = ip.to(DEVICE), target.to(DEVICE)
        output = model(ip)
        closs = criterion(output, target)

        optimizer.zero_grad()
        closs.backward()
        nn.utils.clip_grad_norm_(model.parameters(), ARGS.clip_norm)
        optimizer.step()

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        closses.update(closs.item(), ip.size(0))
        top1.update(prec1.item(), ip.size(0))

        if i % ARGS.print_every == 0:
            print(f"Epoch: {epoch}\tLoss: {closses.avg}\tAccuracy: {top1.avg}")

        # Log to Tensorboard
        if ARGS.tensorboard:
            WRITER.add_scalar("train_loss", closses.avg, epoch)
            WRITER.add_scalar("train_acc", top1.avg, epoch)


def validate(
    val_loader: th.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    epoch: int,
):
    """Model validation step.

    Args:
        val_loader (th.utils.data.DataLoader): Validation loader.
        model (nn.Module): Model object.
        criterion (nn.Module): Loss object.
        epoch (int): CUrrent epoch.
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    for _, (ip, target) in enumerate(val_loader):
        ip, target = ip.to(DEVICE), target.to(DEVICE)
        with th.no_grad():
            output = model(ip)
            loss = criterion(output, target)
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), ip.size(0))
            top1.update(prec1.item(), ip.size(0))

    print(" * Prec@1 {top1.avg:3f}".format(top1=top1))

    if ARGS.tensorboard:
        WRITER.add_scalar("val_loss", losses.avg, epoch)
        WRITER.add_scalar("val_acc", top1.avg, epoch)


class AverageMeter(object):
    """Method to compute the average and current value."""

    def __init__(self):
        """Init function."""
        self.reset()

    def reset(self):
        """Reset values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update values."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Compute the precision@k for the specified values of k.

    Args:
        output (Tensor): Tensor containing the predicted classes
        target (Tensor): Tensor containing the labels
        topk (tuple, optional): Calculate particular k. Defaults to (1,).

    Returns:
        [float]: top k precision calculated value
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
