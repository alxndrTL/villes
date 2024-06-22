"""
Script équivalent à 2_mlp.ipynb.
Utilisé pour lancer rapidement une expé sur wandb.
"""

import wandb
wandb.login()

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = "cuda"

# chargement des données
fichier = open('villes.txt')
donnees = fichier.read()
villes = donnees.replace('\n', ',').split(',')
villes = [ville for ville in villes if len(ville) > 2]

# création du vocabulaire

vocabulaire = sorted(list(set(''.join(villes))))
vocabulaire = ["<pad>", "<SOS>", "<EOS>"] + vocabulaire
# <SOS> et <EOS> sont ajoutés respectivement au début et à la fin de chaque séquence
# <pad> est utilisé pour faire en sorte que toutes les séquences aient la même longueur

# pour convertir char <-> int
char_to_int = {}
int_to_char = {}

for (c, i) in zip(vocabulaire, range(len(vocabulaire))):
    char_to_int[c] = i
    int_to_char[i] = c

num_sequences = len(villes)
max_len = max([len(ville) for ville in villes]) + 2 # <SOS> et <EOS>

X = torch.zeros((num_sequences, max_len), dtype=torch.int32)

for i in range(num_sequences):
    X[i] = torch.tensor([char_to_int['<SOS>']] + [char_to_int[c] for c in villes[i]] + [char_to_int['<EOS>']] + [char_to_int['<pad>']] * (max_len - len(villes[i]) - 2))

n_split = int(0.9*X.shape[0])

idx_permut = torch.randperm(X.shape[0])
idx_train, _ = torch.sort(idx_permut[:n_split])
idx_val, _ = torch.sort(idx_permut[n_split:])

X_train = X[idx_train]
X_val = X[idx_val]

def get_batch(split, batch_size):
    data = X_train if split == 'train' else X_val

    idx_seed = torch.randint(low=int(batch_size/2), high=int(data.shape[0]-batch_size/2), size=(1,), dtype=torch.int32).item()

    batch = data[int(idx_seed-batch_size/2):int(idx_seed+batch_size/2)]
    X = batch[:, :-1].to(device) # (B, L=max_len-1=46)
    Y = batch[:, 1:].to(device) # (B, L)
    return X, Y.long()

class BengioMLP(nn.Module):
    def __init__(self, d_model, d_hidden, n_context, vocabulaire):
        super().__init__()

        self.vocabulaire = vocabulaire
        self.n_context = n_context

        self.embed = nn.Embedding(len(vocabulaire), d_model)

        self.fc1 = nn.Linear(n_context * d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, len(vocabulaire))

    def forward(self, idx):
        embeddings = []
        for _ in range(self.n_context):
            embd = self.embed(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = char_to_int['<SOS>']
            embeddings.append(embd)

        embeddings = torch.cat(embeddings, -1) # (B, L, n_context*d_model)

        x = F.tanh(self.fc1(embeddings)) # (B, L, d_hidden)
        logits = self.fc2(x) # (B, L, vocab_size)

        return logits
    
    def sample(self, prompt = "", g = torch.Generator(), device="cpu"):
        idx = torch.tensor([char_to_int[c] for c in prompt], dtype=torch.int32, device=device).unsqueeze(0)
        idx = torch.cat([torch.tensor(char_to_int['<SOS>'], device=device).view(1, 1), idx], dim=1)
        next_id = -1

        while next_id != char_to_int['<EOS>']:
            idx_cond = idx if idx.size(1) <= self.n_context else idx[:, -self.n_context:]
            logits = self.forward(idx_cond) # (1, l, vocab_size)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=g).item()
            idx = torch.cat([idx, torch.tensor(next_id, device=device).view(1, 1)], dim=1)
        
        return "".join([int_to_char[p.item()] for p in idx[0, 1:-1]])
    
# hyperparamètres
d_model = 32
d_hidden = 512
n_context = 20

iterations = 20000
lr = 3e-4
batch_size = 64

model = BengioMLP(d_model=d_model, d_hidden=d_hidden, n_context=n_context, vocabulaire=vocabulaire).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

num_params = sum(p.numel() for p in model.parameters())
print(f"Nombre de paramètres : {num_params}")

run = wandb.init(
    project="villes",
    config={
        "architecture": "MLP",
        "d_model": d_model,
        "d_hidden": d_hidden,
        "n_context": n_context,
        "learning_rate": lr,
        "iterations": iterations,
        "batch_size": batch_size
    },
)

g = torch.Generator(device).manual_seed(123456789)

for i in range(iterations):
    X, Y = get_batch('train', batch_size) # (B, L)
    logits = model(X) # (B, L, vocab_size)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=char_to_int['<pad>'])
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    to_log = {}
    if i%50==0:
        # loss val
        X, Y = get_batch('val', batch_size) # (B, L)
        logits = model(X) # (B, L, vocab_size)
        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=char_to_int['<pad>'])

        to_log.update({"loss_train": loss.item(), "loss_val": val_loss.item()})
        #print(f"train loss : {loss.item():.2f} | val loss : {val_loss.item():.2f}")

    if i%1000==0:
        # % de noms générés qui existent déjà
        total = 100
        compteur_existants = 0
        for _ in range(total):
            nom = model.sample(g=g, device=device)
            if nom in villes:
                compteur_existants += 1
        
        to_log.update({'prct_existants': compteur_existants/total})

    if to_log:
        wandb.log(to_log, step=i)

to_log = {"num_params": num_params, "num_epochs": (iterations*batch_size)/X_train.shape[0]}
wandb.log(to_log)
wandb.finish()

print("Terminé.")
