import torch as th
from vtab import get_data, get_classes_num
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.models import create_model
from model import CPLoraMerged
from tqdm import tqdm
import torch.nn.functional as F
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import tensorly as tl
tl.set_backend("pytorch")


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
        if ep == 49:
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
    print(args)
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
            bias = b
        )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias if b else None
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "_ft" in name:
            param.requires_grad = True

    # init_fn = th.nn.init.xavier_normal_
    # init_fn = th.nn.init.xavier_uniform_
    # init_fn = th.nn.init.eye_
    # init_fn = th.nn.init.orthogonal_
    # init_fn = th.nn.init.zeros_
    init_fn = "CP"
    print("\n\nUsing CP init\n\n")

    def split_weight(weight):
        dim1, dim2 = weight.shape
        if dim1 != 3 * dim2:
            raise RuntimeError("dim1 != 3 * dim2")
        dim1 //= 3
        layer_weight = weight.reshape(3, dim1, dim2)
        return layer_weight.unbind(0)



    for name, module in model.named_modules():
        if "qkv" in name and isinstance(module, CPLoraMerged):
            if init_fn == "CP":
                decomp = tl.decomposition.CP(rank=args.trainable_ranks, normalize_factors=False, verbose=False, init="random", tol=1e-24, random_state=42)
                Q_weight, K_weight, V_weight = split_weight(module.weight.data)
                tensor_fn = lambda x: x.reshape((x.shape[0], 12, x.shape[0]//12))
                lambdas_Q, (model_Q, heads_Q, headdim_Q) = decomp.fit_transform(tensor_fn(Q_weight))
                lambdas_K, (model_K, heads_K, headdim_K) = decomp.fit_transform(tensor_fn(K_weight))
                lambdas_V, (model_V, heads_V, headdim_V) = decomp.fit_transform(tensor_fn(V_weight))
                module.CPQ.S_model_ft = th.nn.Parameter(model_Q) 
                module.CPK.S_model_ft = th.nn.Parameter(model_K) 
                module.CPV.S_model_ft = th.nn.Parameter(model_V)
                module.CPQ.S_heads_ft = th.nn.Parameter(heads_Q) 
                module.CPK.S_heads_ft = th.nn.Parameter(heads_K) 
                module.CPV.S_heads_ft = th.nn.Parameter(heads_V)
                module.CPQ.S_headdim_ft = th.nn.Parameter(headdim_Q) 
                module.CPK.S_headdim_ft = th.nn.Parameter(headdim_K) 
                module.CPV.S_headdim_ft = th.nn.Parameter(headdim_V)
                # module.CPQ.lambdas_ft = th.nn.Parameter(lambdas_Q)
                # module.CPK.lambdas_ft = th.nn.Parameter(lambdas_K)
                # module.CPV.lambdas_ft = th.nn.Parameter(lambdas_V)


            else:
                init_fn(module.CPQ.S_model_ft)
                init_fn(module.CPK.S_model_ft)
                init_fn(module.CPV.S_model_ft)
                init_fn(module.CPQ.S_heads_ft)
                init_fn(module.CPK.S_heads_ft)
                init_fn(module.CPV.S_heads_ft)
                init_fn(module.CPQ.S_headdim_ft)
                init_fn(module.CPK.S_headdim_ft)
                init_fn(module.CPV.S_headdim_ft)
                # th.nn.init.normal_(module.CPQ.lambdas_ft)
                # th.nn.init.normal_(module.CPK.lambdas_ft)
                # th.nn.init.normal_(module.CPV.lambdas_ft)


            print(th.norm(module.weight))
            print(th.norm(module.CPQ.S_model_ft), th.norm(module.CPQ.S_heads_ft), th.norm(module.CPQ.S_headdim_ft))
            print(th.norm(module.CPK.S_model_ft), th.norm(module.CPK.S_heads_ft), th.norm(module.CPK.S_headdim_ft))
            print(th.norm(module.CPV.S_model_ft), th.norm(module.CPV.S_heads_ft), th.norm(module.CPV.S_headdim_ft))
            print()

    model.reset_classifier(num_classes)
    model = th.nn.DataParallel(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = round(trainable_params / 1000000, 4)
    print(f"Trainable #Parameters: {trainable_params}M")

    opt = th.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # opt = th.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    # scheduler = CosineLRScheduler(opt, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    scheduler = None
    model = train(model, train_dl, test_dl, opt, scheduler, 100)
    facc = test(model, test_dl)
    print(f"Final Accuracy: {facc}")

if __name__ == "__main__":
    main()
