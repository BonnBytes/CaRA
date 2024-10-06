import torch as th
from torch import nn
from tqdm import tqdm 
import numpy  as np
import random
import _testimportmultiple
from timm.models import create_model
from argparse import ArgumentParser
from vtab import *
import yaml
import timm


def train(args, model, dl, opt, sched, epochs):
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

    model = model.cpu()
    return model


@th.no_grad()
def test(model, dl):
    model.eval()
    model = model.cuda()
    for batch in dl:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
    # Add accuracy calculation here


def cp_attn(self, x):
    pass

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
            s            
