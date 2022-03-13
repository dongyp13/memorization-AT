# Exploring Memorization in Adversarial Training

This repository contains the code for "*Exploring Memorization in Adversarial Training*", submitted to ICLR 2022.

## Prerequisites
* Python (3.6.8)
* Pytorch (1.3.0)
* torchvision (0.4.1)
* numpy

## Training with Random Labels

### On CIFAR-10


For PGD-AT

```
python train.py --wd 0 --noise-type label_symmetric --noise-rate 1.0
```

For TRAEDS

```
python train.py --wd 0 --noise-type label_symmetric --noise-rate 1.0 --loss-type trades
```

## Overcome Robust Overfitting

### On CIFAR-10

For PGD-AT + TE

```
python train_te.py
```

For TRADES + TE

```
python train_te.py --loss-type trades
```

## Pre-trained Models
We will release the pretrained models after the review process.