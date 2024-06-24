"""
Script équivalent à 4_transformer.ipynb.
Utilisé pour lancer rapidement une expé sur wandb.
"""

import os
os.chdir('/workspace/')

import wandb
wandb.login()

import torch
import torch.nn.functional as F

from mamba.mamba import MambaConfig
from lm import LM

from data import Dataset

device = "cuda"

# hyperparamètres
d_model = 512
n_layers = 1
dropout = 0.

iterations = 40000
lr = 3e-4
batch_size = 32 # mamba.py est gourmand en RAM, ne pas hésiter à baisser (sinon passer à l'implémentation CUDA avec use_cuda=True)

dataset = Dataset(device=device) # toute la partie données de 2_mlp.py a été encapsulée dans l'objet Dataset

config = MambaConfig(d_model=d_model, n_layers=n_layers, use_cuda=True)
model = LM(config, vocab_size=len(dataset.vocabulaire)).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

num_params = sum(p.numel() for p in model.parameters())
print(f"Nombre de paramètres : {num_params}")

run = wandb.init(
    project="villes",
    config={
        "architecture": "Mamba",
        "d_model": d_model,
        "n_layers": n_layers,
        "dropout": dropout,
        "learning_rate": lr,
        "iterations": iterations,
        "batch_size": batch_size
    },
)

g = torch.Generator(device).manual_seed(123456789)

for i in range(iterations):
    X, Y = dataset.get_batch('train', batch_size) # (B, L)
    logits = model(X) # (B, L, vocab_size)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=dataset.char_to_int['<pad>'])
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    to_log = {}
    if i%50==0:
        # loss val
        X, Y = dataset.get_batch('val', batch_size) # (B, L)
        logits = model(X) # (B, L, vocab_size)
        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=dataset.char_to_int['<pad>'])

        to_log.update({"loss_train": loss.item(), "loss_val": val_loss.item()})
        #print(f"train loss : {loss.item():.2f} | val loss : {val_loss.item():.2f}")

    if to_log:
        wandb.log(to_log, step=i//2) # x2 pour pouvoir comparer avec les MLPs sur wandb

    if i%1000==0:
        print(i)

to_log = {"num_params": num_params, "num_epochs": (iterations*batch_size)/dataset.X_train.shape[0]}
wandb.log(to_log)
wandb.finish()

print("Terminé.")

torch.save(model.state_dict(), f"mamba_d_model{d_model}.pth")