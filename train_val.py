import torch as th
from vtab import get_data, get_classes_num
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.models import create_model
from model import CPLoraMerged
from tqdm import tqdm
import torch.nn.functional as F
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def train(model, dl, tdl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    for ep in (pbar:=tqdm(range(epoch))):
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep%10 == 50:
            acc = test(model, tdl)
            pbar.set_description(f"Accuracy: {str(acc)}")
    model = model.cpu()
    return model

@th.no_grad()
def test(model, dl):
    model.eval()
    acc = []
    ex = []
    model = model.cuda()
    for batch in dl:
        x, y =  batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        out = th.argmax(out, dim=1)
        correct = (out == y).float()
        acc.append(correct.sum())
        ex.append(len(y))
    ac = sum(acc) / sum(ex)
    return round(ac.cpu().item(), 4)


def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--trainable-ranks",
        default=2,
        type=int,

        help="Number of trainable ranks."
    )
    return parser.parse_args()


def main():
    th.manual_seed(42)
    args = _parse_args()
    name = "svhn"
    train_dl, test_dl = get_data(name)
    num_classes = get_classes_num(name)
    
    model = create_model("vit_base_patch16_224_in21k", checkpoint_path="./ViT-B_16.npz", drop_path_rate=0.1)    
    qkv_layers = []
    for name, _ in model.named_modules():
        if "qkv" in name:
            qkv_layers.append(name.split("."))
    for idx, (*parent, k, _, _) in enumerate(qkv_layers):
        nm = '.'.join(qkv_layers[idx])
        mod = model.get_submodule(nm)
        b = True if mod.bias is not None else False
        new_mod = CPLoraMerged(
            in_features=768,
            out_features=768*3,
            tr_rank=args.trainable_ranks,
            embed_dim=768,
            num_heads=12,
            bias = True
        )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "_ft" in name:
            param.requires_grad = True    

    model.reset_classifier(num_classes)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = round(trainable_params / 1000000, 4)
    print(f"Trainable #Parameters: {trainable_params}M")

    opt = th.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineLRScheduler(opt, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    # scheduler = None
    model = train(model, train_dl, test_dl, opt, scheduler, 100)
    facc = test(model, test_dl)
    print(f"Final Accuracy: {facc}")

if __name__ == "__main__":
    main()