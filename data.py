import torch

class Dataset():
    def __init__(self, file_name: str = "villes.txt"):

        # chargement des données
        fichier = open(file_name)
        donnees = fichier.read()
        villes = donnees.replace('\n', ',').split(',')
        villes = [ville for ville in villes if len(ville) > 2]

        # création du vocabulaire

        self.vocabulaire = sorted(list(set(''.join(villes))))
        self.vocabulaire = ["<pad>", "<SOS>", "<EOS>"] + self.vocabulaire
        # <SOS> et <EOS> sont ajoutés respectivement au début et à la fin de chaque séquence
        # <pad> est utilisé pour faire en sorte que toutes les séquences aient la même longueur

        # pour convertir char <-> int
        self.char_to_int = {}
        self.int_to_char = {}

        for (c, i) in zip(self.vocabulaire, range(len(self.vocabulaire))):
            self.char_to_int[c] = i
            self.int_to_char[i] = c


        num_sequences = len(villes)
        self.max_len = max([len(ville) for ville in villes]) + 2 # <SOS> et <EOS>

        X = torch.zeros((num_sequences, self.max_len), dtype=torch.int32)

        for i in range(num_sequences):
            X[i] = torch.tensor([self.char_to_int['<SOS>']] +
                                [self.char_to_int[c] for c in villes[i]] +
                                [self.char_to_int['<EOS>']] +
                                [self.char_to_int['<pad>']] * (self.max_len - len(villes[i]) - 2))

        n_split = int(0.9*X.shape[0])

        idx_permut = torch.randperm(X.shape[0])
        idx_train, _ = torch.sort(idx_permut[:n_split])
        idx_val, _ = torch.sort(idx_permut[n_split:])

        self.X_train = X[idx_train]
        self.X_val = X[idx_val]

    def get_batch(self, split, batch_size):
        data = self.X_train if split == 'train' else self.X_val

        idx = torch.randint(low=int(batch_size/2), high=int(data.shape[0]-batch_size/2), size=(1,), dtype=torch.int32).item()

        batch = data[int(idx-batch_size/2):int(idx+batch_size/2)]
        X = batch[:, :-1] # (B, L=max_len-1=46)
        Y = batch[:, 1:] # (B, L)
        return X, Y.long()
                    