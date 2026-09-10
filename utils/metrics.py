# ---------------------------------------
# BLOC 1 - Imports 
# ---------------------------------------
# utils/metrics.py - Les métriques Dice & ASSD (≈ utils/EvaluationHelper.py de PetSeg)
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

# ---------------------------------------
# BLOC 2 - le Dice 
# ---------------------------------------
def dice(pred, vrai):
    # pred et vrai des tableaux de vrai/faux (True là où c'est la classe
    pred = pred.astype(bool)
    vrai = vrai.astype(bool)
    total = pred.sum() + vrai.sum()
    if total == 0:          # les deux vides -> on considère parfait
        return 1.0 
    return 2.0 * (pred & vrai).sum() / total

# ---------------------------------------
# BLOC 3 - le Dice 
# ---------------------------------------
def _surface(m):
    # le contour d'une forme = la forme moins sa version rognée d'un pixel
    return m & ~binary_erosion(m)

def assd(pred, vrai):
    pred = pred.astype(bool)
    vrai = vrai.astype(bool)
    if pred.sum() == 0 or vrai.sum == 0:
        return float("nan")         # impossible à définir si une forme est vide 
    sp, sv = _surface(pred), _surface(vrai)
    d1 = distance_transform_edt(~sv)[sp]     # distance de chaque point du contout pred au contour vrai
    d2 = distance_transform_edt(~sv)[sv]     # et l'inverse
    return float((d1.sum() + d2.sum()) / (len(d1) + len(d2)))     # moyenne des deux

if __name__ == "__main__":
    vrai = np.zeros((20, 20), dtype=bool); vrai[5:15, 5:15] = True  # un carré
    dec = np.zeros((20, 20), dtype=bool); dec[7:17, 7:17] = True    # le même décalé
    print("parfait -> dice", round(dice(vrai, vrai), 3), "| assd", round(assd(vrai, vrai), 3))
    print("décalé -> dice", round(dice(dec, vrai), 3), "| assd", round(assd(dec, vrai), 3))