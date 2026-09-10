# MPetSeg

A clean, simplified reconstruction of the PetSeg semantic segmentation
project, built for educational purposes to understand each component
of a PyTorch segmentation pipeline.

## Overview

PetSeg performs supervised semantic segmentation on the Oxford-IIIT Pet
dataset, classifying each pixel into three classes: animal (foreground),
background, and boundary (not-classified).

This reconstruction rewrites the core pipeline component by component,
in a simplified and heavily commented form, while preserving the same
techniques as the original project.

## Structure

- `data/dataset.py` — loads, resizes, normalizes and augments the images
  and masks (Oxford-IIIT Pet).
- `model.py` — a U-Net with a frozen, ImageNet-pretrained MobileNetV3
  encoder and a decoder trained from scratch (transfer learning).
- `utils/metrics.py` — the Dice coefficient and ASSD evaluation metrics.
- `train.py` — the training loop (forward, loss, backward, weights update),
  training only the decoder.
- `visualize.py` — saves prediction triptychs (image / true mask / predicted mask).

## Techniques covered

- ImageNet normalization (MEAN / STD)
- Trimap label handling (1, 2, 3 → 0, 1, 2)
- Nearest-neighbor resizing for masks
- Transfer learning with a frozen pretrained encoder
- U-Net skip connections (feature concatenation)
- Dice and ASSD segmentation metrics

## Requirements

- Python 3.10
- PyTorch, torchvision, opencv-python, scipy, pillow, matplotlib

## Usage

```bash
python train.py        # train the model (weights saved to poids.pth)
python visualize.py    # generate prediction images
```

## Note

This is a personal reconstruction for learning purposes, based on the
original PetSeg project. The Oxford-IIIT Pet dataset is downloaded
automatically on first run and is not included in this repository.
Also various learning resources were used, including documentation, courses, and an AI assistant as a tutor.

## Acknowledgements

Original project: PetSeg (Imaging-based Computational Biomedicine
Laboratory, Nara Institute of Science and Technology).
