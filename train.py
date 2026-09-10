# ------------------------------
# BLOC 1 : Les Imports
# ------------------------------
# train.py - la boucle d'entraînement (≈ train.py de PetSeg)
# C'est le chef d'orchestre : il IMPORTE les autres fichiers et les fait travailler ensemble 
import torch 
from torch.utils.data import DataLoader 
from torch.optim import Adam

from data.dataset import PetDataset # notre brique 1 (les données)
from model import UNet  # notre brique 2 (le U-Net)
from utils.metrics import dice  # notre brique 3 (la métrique)

# -------------------------------------
# BLOC 2 : Préparer les ingrédients
# -------------------------------------
def main():
    # 1) les données, découpées en lots 
    ds = PetDataset("work_space/data", entrainement=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    
    # 2) le modèle 
    model = UNet(n_classes=3)
    
    # 3) l'optimiseur(SEULEMENT le décodeur -> encodeur gelé !) et la perte
    opt = Adam(model.decoder.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    
    # -------------------------------------
    # BLOC 3 : Préparer les ingrédients
    # -------------------------------------
    
    # 4) la boucle : pour chaque époque, pour chaque lot -> les 4 étapes
    for epoch in range(1, 5):       # 2 époques (petit test)
        model.train()
        total = 0
        for images, masques in dl:
            pred = model(images)                # 1. forward    : la prédiction
            loss = crit(pred, (masques).long()) # 2. perte      : à quel point on se trompe # .long() = force le masque en entiers (int64) # (masques - 1 si erreur )
            opt.zero_grad()                     #    Remise à zéro des gradients
            loss.backward()                     # 3. backward   : calcul des gradients
            opt.step()                          # 4. mise à jour des poids du décodeur
            total += loss.item()        
        print(f"epoch {epoch} : perte moyenne = {total / len(dl):.3f}")
    
    # -------------------------------------
    # BLOC 4 : sauvegarder & lancer
    # -------------------------------------
    # 5) sauvegarder les poids appris
    torch.save(model.state_dict(), "poids.pth")
    print("poids sauvegardés dans poids.pth")
    

if __name__ == "__main__":
    main()
    
        