# ---------------------------------------
# BLOC 1 : Imports + Encodeur
# ---------------------------------------
# model.py -- le U-Net : Encodeur pré-entraîné (gelé) + décodeur (entraîné)
# Ce premier Bloc contient les imports et la définition de l'encodeur (MobileNetV3 pré-entraîné et gelé) ainsi que les paramètres pour le décodeur.
# La Classe UNet hérite de nn.Module, ce qui est standard pour les modèles PyTorch. Le constructeur initialise l'encodeur et définit les indices des couches pour les connexions de saut ainsi que le nombre de canaux pour chaque connexion de saut et pour chaque étage du décodeur.
# nn.Module est la classe de base pour tous les modules de réseau de neurones dans PyTorch. En héritant de cette classe, UNet peut utiliser toutes les fonctionnalités de PyTorch pour la construction et l'entraînement du modèle.
# self.encodeur est initialisé avec MobileNetV3 pré-entraîné, et les couches de l'encodeur sont gelées pour éviter qu'elles ne soient mises à jour pendant l'entraînement. Les indices des couches pour les connexions de saut et le nombre de canaux pour chaque connexion de saut sont définis en fonction de l'architecture de MobileNetV3.
# L'encodage c'est la partie du réseau qui extrait les caractéristiques de l'image d'entrée, et le décodeur est la partie qui reconstruit l'image de sortie à partir de ces caractéristiques. Les connexions de saut permettent de combiner les informations des couches profondes avec celles des couches plus superficielles, ce qui aide à préserver les détails spatiaux dans l'image reconstruite.
# Alors comment l'encodage marche en terme simple c'est que l'image d'entrée passe par plusieurs couches convolutives qui extraient des caractéristiques de plus en plus complexes. Chaque couche apprend à détecter différents aspects de l'image, comme les bords, les textures et les formes. Les connexions de saut permettent de conserver certaines informations des couches précédentes pour aider à reconstruire l'image finale avec plus de détails.
# Quand on parle des caracteristiques extraites par l'encodeur, on fait référence aux représentations internes de l'image que le réseau apprend à chaque couche. Ces caractéristiques sont des cartes de caractéristiques (feature maps) qui capturent différents aspects de l'image, comme les contours, les textures et les motifs. Ces cartes de caractéristiques sont ensuite utilisées par le décodeur pour reconstruire l'image de sortie.
import torch 
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__() 
        
        # Encodeur : MobileNetV3 pré-entraîné qu'on GÈLE
        # .feautures : La partie qui extrait les caractéristiques de l'image
        self.encoder = models.mobilenet_v3_large(weights = "DEFAULT").features
        
        # Les couches de l'encodeur d'où on prélève les "connexions de saut"
        self.skip_ids = [1, 3, 6, 12, 16]  # indices des couches de l'encodeur pour les connexions de saut
        # Ses valeurs sont déterminées par l'architecture de MobileNetV3 et les dimensions des cartes de caractéristiques à chaque étape.
        # On les retrouve en inspectant la structure du modèle MobileNetV3 et en notant les dimensions des sorties de chaque bloc.
        skip_ch = [16, 24, 40, 112, 960] # nombre de canaux pour chaque connexion de saut
        # Ses valeurs sont également déterminées par l'architecture de MobileNetV3 et correspondent au nombre de canaux de sortie des couches spécifiées dans skip_ids.
        # ...et nombre de canaux voulu à chaque étage du décodeur (On s'en sert au Bloc 2)
        up_ch = [n_classes, 64, 128, 256, 512, 0]
        
        # ---------------------------------------
        # BLOC 1 : Imports + Encodeur
        # --------------------------------------- 
        # --- DECODER : 5 etages qui remontent (du plus profond vers la sortie) ---
        blocs = []
        for i in reversed(range(5)) : # i = 4, 3, 2, 1, 0
            entree = skip_ch[i] + up_ch[i + 1] # canaux en entrée = saut + remontée précédent
            sortie = up_ch[i]                   # canaux en sortie de cet étage du décodeur
            if i == 0 : 
                # dernie étage : produit directement les n_classes canaux du masque
                blocs.append(nn.ConvTranspose2d(entree, sortie, 3, 2, 1, output_padding=1))
            else : 
                # étage normal : agrandir (ConvTranspose) + stabiliser (BatchNorm) + activer (ReLU)
                blocs.append(nn.Sequential(
                    nn.ConvTranspose2d(entree, sortie, 3, 2, 1, output_padding=1, bias=False),
                    nn.BatchNorm2d(sortie),
                    nn.ReLU(inplace=True)))
        self.decoder = nn.ModuleList(blocs)
        print(blocs)
                
    def forward(self, x):
        # 1) DESCENTE : on passe dans l'encodeur et on met de côté les "sauts"
        sauts = []
        for i, couche in enumerate(self.encoder):
            x = couche(x)
            if i in self.skip_ids:
                sauts.append(x)
                
        # on inverse : le décodeur remonte du plus profond au plus fin
        sauts = sauts[::-1]
        
        # 2) REMONTÉE : Premier étage sur le saut le plus profond... 
        y = self.decoder[0](sauts[0])
        # ...puis à chaque étage, on CONCATENE avec le saut correspondant (le "U")
        for i in range(1, len(self.decoder)):
            y = self.decoder[i](torch.cat((y, sauts[i]), dim=1))
            
        return y        # (lot, n_classes, 224, 224) : un score par classe et par pixel
    
    
            
if __name__ == "__main__" : 
    model = UNet(n_classes=3)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("entrée :", tuple(x.shape), "-> sortie :", tuple(y.shape))
                