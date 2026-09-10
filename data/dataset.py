# ---------------------------------------
# BLOC 1
# ---------------------------------------

import os, glob
import numpy as np
import torch
import cv2
from PIL import Image 
from torch.utils.data import Dataset 

# Normalization ImageNet : même chiffres que PetSeg (l'encodeur a appris avec ces chiffres)
# Pour rappel, la normalisation est faite avec la formule : (image - mean) / std
# image est un tableau numpy de type float32, avec des valeurs entre 0 et 1. 
# Ces valeurs sont obtenues en divisant les valeurs de l'image (0-255) par 255.0
# On normalise pas le masque, car il contient des valeurs discrètes (0, 1, 2) qui représentent les classes.
# Le masque c'est l'image de sortie que le modèle doit prédire, donc on ne veut pas la normaliser.
# On normalise l'image pour que le modèle puisse mieux apprendre les caractéristiques des images, mais on ne normalise pas le masque pour que le modèle puisse apprendre à prédire les classes correctement.
# Les avantages de la normalisation sont :
# - Les valeurs des pixels sont centrées autour de 0, ce qui permet au modèle d'apprendre plus facilement les caractéristiques des images.
# Pour Mean et Std, on utilise les valeurs de ImageNet car le modèle a été pré-entraîné sur ce dataset.
# Mean : 0.485, 0.456, 0.406 viennent de la moyenne des canaux R, G et B des images de ImageNet : Autrement dit, si on prend toutes les images de ImageNet et qu'on calcule la moyenne des valeurs de chaque canal (R, G, B), on obtient ces valeurs.
# Std : 0.229, 0.224, 0.225 viennent de l'écart type des canaux R, G et B des images de ImageNet : Autrement dit, si on prend toutes les images de ImageNet et qu'on calcule l'écart type des valeurs de chaque canal (R, G, B), on obtient ces valeurs.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Les 3 classes du masque, après le décalage -1
# decalage -1 : on soustrait 1 à toutes les valeurs du masque, pour que les classes soient 0, 1 et 2 au lieu de 1, 2 et 3.
# Ça veut dire en terme simple que le masque est codé avec 3 valeurs : 0, 1 et 2.
# 0 : "Animal/pet", 1 : "Fond/background", 2 : "Contour/edge"
# J'ai noté 2 : "Contour/edge". C'est presque ça. Dans Oxford Pet, la classe 2 s'appelle « Not-classified » 
# — c'est la fine bande autour de l'animal,  une zone incertaine que les annotateurs n'ont pas voulu trancher 
# (ni franchement animal, ni franchement fond). « Contour » est une bonne image 
# mentale, garde-la, mais sache que le nom officiel veut dire « zone indécise », pas « bord net ». 
# (Souviens-toi : c'est justement cette classe qui avait le moins bon Dice dans tes résultats — 
# maintenant tu sais pourquoi : elle est fine et ambiguë par nature.)
NOMS_CLASSES = {0: "Animal", 1: "Fond", 2: "Contour"}


# ---------------------------------------
# BLOC 2
# ---------------------------------------
# Detaillons le Bloc 2 : la classe PetDataset. 
# C'est une sous-classe de Dataset, qui est une classe abstraite de PyTorch.
# La classe Dataset de PyTorch est une classe qui permet de créer des datasets personnalisés.
# Cette classe est utilisée pour charger les images et les masques du dataset Oxford-IIIT Pet.
# from torch.utils.data import Dataset : on importe la classe Dataset de PyTorch.
# (Classe dans Pytorch) : https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# On doit implémenter les méthodes __len__ et __getitem__ pour que notre dataset soit compatible avec PyTorch.

class PetDataset(Dataset):
    def __init__(self, data_root, taille=(224, 224), entrainement=True):
        self.taille = taille
        self.entrainement = entrainement
        d = os.path.join(data_root, "oxford-iiit-pet")
        self.images = sorted(glob.glob(os.path.join(d, "images", "*.jpg")))
        self.masques = [os.path.join(d, "annotations", "trimaps", 
                                     os.path.basename(p)[:-4] + ".png") for p in self.images]
        
    def __len__(self):
        return len(self.images)
    
    # ---------------------------------------
    # BLOC 3
    # ---------------------------------------
    # Ce bloc contient la méthode __getitem__ de la classe PetDataset.
    # Cette méthode est appelée par PyTorch pour récupérer un élément du dataset.
    # Elle prend en entrée un index i et retourne l'image et le masque correspondants.
    # Donc pour charger un batch d'images et de masques, PyTorch va appeler cette méthode pour chaque index du batch.
    # Un batch est un ensemble d'images et de masques qui sont traités en parallèle par le modèle.
    # On convertit images en RGB et masques en niveaux de gris (1 canal) pour que le modèle puisse apprendre à prédire les classes correctement.
    # Image en RGB pour que le modèle puisse apprendre les caractéristiques des images en couleur, et masque en niveaux de gris pour que le modèle puisse apprendre à prédire les classes correctement.
    # On redimensionne parce que les images et les masques du dataset Oxford-IIIT Pet ont des tailles différentes, et que le modèle attend des images et des masques de taille fixe.
    # Le Trimap est une image qui contient les classes de l'image : 0 pour l'animal, 1 pour le fond et 2 pour le contour. On soustrait 1 à toutes les valeurs du masque pour que les classes soient 0, 1 et 2 au lieu de 1, 2 et 3.
    # C'est bien de soustraire 1 à toutes les valeurs du masque pour que les classes soient 0, 1 et 2 au lieu de 1, 2 et 3, car ça permet de simplifier le code et d'éviter des erreurs de décalage d'index.
    # On applique des transformations aléatoires sur l'image et le masque pour augmenter la diversité des données. C'est ce qu'on appelle l'augmentation de données (data augmentation). Cela permet au modèle d'apprendre à généraliser et de ne pas sur-apprendre les caractéristiques des images du dataset.
    # Et pour cela, on utilise des transformations simples : retournement horizontal et vertical. On pourrait utiliser d'autres transformations comme la rotation, le zoom, le changement de luminosité, etc. Mais pour l'instant, on se limite à ces deux transformations.
    # Et ensuite on normalise l'image pour que le modèle puisse mieux apprendre les caractéristiques des images, mais on ne normalise pas le masque pour que le modèle puisse apprendre à prédire les classes correctement.
    # On passe ensuite l'image et le masque en tenseur PyTorch pour que le modèle puisse les utiliser. On utilise la méthode torch.from_numpy pour convertir un tableau numpy en tenseur PyTorch.
    # Si c'est en tableau numpy, on utilise la méthode transpose pour changer l'ordre des dimensions de l'image : (H, W, C) -> (C, H, W). Cela permet au modèle de traiter les images correctement.
    # On utilise la méthode copy pour créer une copie de l'image et du masque, car sinon PyTorch pourrait modifier les données originales. Cela permet d'éviter des erreurs de modification des données originales.
    # A partir de là, on travaille la copie de l'image et du masque, donc on peut les modifier sans risque.
    # A la difference des tableaux numpy, les tenseurs PyTorch sont des objets qui permettent de faire des calculs sur GPU. C'est pour ça qu'on convertit les tableaux numpy en tenseurs PyTorch. 
    def __getitem__(self, i):
        # 1 : CHARGEMENT de l'image et du masque (fichier -> tableau numpy)
        image = np.array(Image.open(self.images[i]).convert("RGB"))
        masque = np.array(Image.open(self.masques[i]))
        # RREDIMENSIONNEMENT edimensionnement de l'image et du masque à la taille souhaitée : image en dimensions (H, W, C) et masque en dimensions (H, W)
        # Cela veut dire que l'image est un tableau 3D avec les dimensions hauteur, largeur et canaux (R, G, B), et que le masque est un tableau 2D avec les dimensions hauteur et largeur.
        # 2 : Redimensionnement de l'image et du masque à la taille souhaitée
        image = cv2.resize(image, self.taille, interpolation = cv2.INTER_LINEAR)
        masque = cv2.resize(masque, self.taille, interpolation = cv2.INTER_NEAREST)
        # 3 : TRIMAP : on soustrait 1 à toutes les valeurs du masque pour que les classes soient 0, 1 et 2 au lieu de 1, 2 et 3.
        masque = masque.astype(np.int64) - 1
        # 4 : AUGMENTATION : si on est en mode entraînement, on applique des transformations aléatoires sur l'image et le masque pour augmenter la diversité des données.
        if self.entrainement:
            if np.random.rand() < 0.5 : image = image[:, ::-1]; masque = masque[:, ::-1]
            if np.random.rand() < 0.5 : image = image[::-1, :]; masque = masque[::-1, :]
        # 5 : NORMALISATION + Passer en tenseur (C, H, L)
        image = (image.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()),
        torch.from_numpy(masque.copy()))
            

# ---------------------------------------
# BLOC 4
# ---------------------------------------
# Le dernier bloc contient le code de test de la classe PetDataset.
# Il nous permet de vérifier que la classe fonctionne correctement en chargeant une image et un masque du dataset et en affichant leurs dimensions et les classes présentes dans le masque.
# ds = PetDataset("work_space/data", entrainement=True) : on crée une instance de la classe PetDataset en lui passant le chemin du dataset et le mode entraînement.
# Si on met entrainement=False, on ne fait pas d'augmentation de données.
# img, msk = ds[0] : on récupère la première image et le premier masque du dataset.
# tuple(img.shape) : on affiche les dimensions de l'image (C, H, W).
# tuple(msk.shape) : on affiche les dimensions du masque (H, W).
# sorted(set(msk.flatten().tolist())) : on affiche les classes présentes dans le masque. On utilise set pour ne garder que les classes uniques, flatten pour aplatir le tableau 2D en 1D, et sorted pour trier les classes par ordre croissant.
if __name__ == "__main__":
    ds = PetDataset("work_space/data", entrainement=True)
    img, msk = ds[0]
    print("images :", len(ds), "| image", tuple(img.shape), "| masque", tuple(msk.shape))
    print("classes :", sorted(set(msk.flatten().tolist())))
    # print(msk)
    

        
    
    
