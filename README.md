# Exploring Memorization in Adversarial Training

This repository contains the code for the following paper

[Exploring Memorization in Adversarial Training](https://openreview.net/forum?id=7gE9V9GBZaI) (ICLR 2022)

[Yinpeng Dong](http://ml.cs.tsinghua.edu.cn/~yinpeng), Ke Xu, [Xiao Yang](http://ml.cs.tsinghua.edu.cn/~xiaoyang), [Tianyu Pang](http://ml.cs.tsinghua.edu.cn/~tianyu), [Zhijie Deng](http://ml.cs.tsinghua.edu.cn/~zhijie),  [Hang Su](http://www.suhangss.me), and [Jun Zhu](http://ml.cs.tsinghua.edu.cn/~jun/index.shtml)

### Citation
If you find our methods useful, please consider citing:

	@inproceedings{
    dong2022exploring,
    title={Exploring Memorization in Adversarial Training},
    author={Yinpeng Dong and Ke Xu and Xiao Yang and Tianyu Pang and Zhijie Deng and Hang Su and Jun Zhu},
    booktitle={International Conference on Learning Representations},
    year={2022}
  }

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
