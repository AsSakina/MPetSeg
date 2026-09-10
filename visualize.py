# -------------------------------------
# BLOC 1 : Imports 
# -------------------------------------
# visualize.py - voir les predictions du modele (≈ eval_and_visual.py de PetSeg)
import torch 
import numpy as np
import matplotlib.pyplot as plt 
import os

from data.dataset import PetDataset, MEAN, STD
from model import UNet

# --------------------------------------
# BLOC 2 : Recharger le modèle entraîné 
# --------------------------------------
# le jeu de données SANS augmentation (on veut voir les vraies images)
ds = PetDataset("work_space/data", entrainement=False)

# on reconstruit le modèle et on remet les poids appris 
model = UNet(n_classes=3)
model.load_state_dict(torch.load("poids.pth"))
model.eval()       # mode évaluation(pas d'entraînement)

# ----------------------------------------
# BLOC 3 : Prédire et Afficher un trypique
# ----------------------------------------
def montrer(i, dossier="resultats"):
    image, masque = ds[i]                       # une image + son vrai masque
    
    with torch.no_grad():                       # pas besoin de gradients pour juste prédire 
        sortie = model(image.unsqueeze(0))      # unsqueeze(0) : ajoute la dimension "lot"
    predit = torch.argmax(sortie, dim=1)[0]     # la classe gagnante à chaque pixel
    
    # dé-normaliser l'image pour l'afficher en vrai couleurs 
    img = image.numpy().transpose(1, 2, 0) * STD + MEAN
    img = img.clip(0, 1)
    
    plt.figure (figsize=(9, 13))
    plt.subplot(1, 3, 1); plt.title("Image");           plt.imshow(img);                plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Masque vrai");     plt.imshow(masque);             plt.axis("off")
    plt.subplot(1, 3, 3,); plt.title("Prédit");         plt.imshow(predit.numpy());     plt.axis("off")
    # plt.show() # Si on garde cette ligne, ça fera qu'on aie des images toutes blanches, vides
    
    
    os.makedirs(dossier, exist_ok=True)             # crée le dossier s'il n'existe pas encore
    plt.savefig(f"{dossier}/resultat_{i}.png", dpi=120, bbox_inches="tight")    # sauvegarde
    plt.close()                 # ferme la figure (économise la mémoire)
    print(f"enregistré : {dossier}/resultat_{i}.png")
    
    
if __name__ == "__main__":
    for i in [0, 1, 2]:         # 3 images d'exemple
        montrer(i)