# evaluer.py — calcule Dice & ASSD par classe sur le jeu de test
import torch
import numpy as np
from data.dataset import PetDataset, NOMS_CLASSES
from model import UNet
from utils.metrics import dice, assd


ds = PetDataset("work_space/data", entrainement=False)   # test = sans augmentation
model = UNet(n_classes=3)
model.load_state_dict(torch.load("poids.pth"))           # recharge tes poids appris
model.eval()

# on prépare un "carnet de notes" : une liste de Dice et d'ASSD pour chaque classe
scores_dice = {c: [] for c in NOMS_CLASSES}   # {0: [], 1: [], 2: []}
scores_assd = {c: [] for c in NOMS_CLASSES}

N = 50   # on évalue sur 50 images (assez pour une moyenne fiable, et pas trop long sur CPU)

with torch.no_grad():
    for i in range(N):
        image, masque = ds[i]
        sortie = model(image.unsqueeze(0))          # prédiction
        predit = torch.argmax(sortie, dim=1)[0].numpy()   # classe gagnante par pixel
        vrai   = masque.numpy()

        # pour chaque classe, on compare "prédit == classe" vs "vrai == classe"
        for c in NOMS_CLASSES:
            p = (predit == c)      # zone prédite de la classe c (vrai/faux)
            v = (vrai == c)        # zone réelle de la classe c
            scores_dice[c].append(dice(p, v))
            a = assd(p, v)
            if not np.isnan(a):    # on ignore les cas où une zone est vide
                scores_assd[c].append(a)
                
                
print(f"Résultats sur {N} images :\n")
for c, nom in NOMS_CLASSES.items():
    d = np.mean(scores_dice[c])
    a = np.mean(scores_assd[c]) if scores_assd[c] else float("nan")
    print(f"  {nom:15s} -> Dice = {d:.3f}   ASSD = {a:.2f} px")
    
