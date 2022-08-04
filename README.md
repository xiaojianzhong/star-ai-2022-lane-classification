# Star AI 2022 Lane Classification

Read this in other languages: English | [简体中文](README_zh-CN.md)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Configuration Name Format](#configuration-name-format)
- [Train](#train)
- [Submit (After Training)](#submit)

## <a name="prerequisites"></a> Prerequisites

- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pillow](https://python-pillow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [torch](https://pytorch.org/)
- [torchinfo](https://github.com/tyleryep/torchinfo)
- [torchsampler](https://github.com/ufoym/imbalanced-dataset-sampler)
- [torchvision](https://pytorch.org/vision/)
- [tqdm](https://github.com/tqdm/tqdm)
- [yacs](https://github.com/rbgirshick/yacs)

All these Python third-party packages can be easily installed through `pip`:

```shell
$ pip install -r requirements.txt
```

## <a name="configuration-name-format"></a> Configuration Name Format

```
{dataset}_{method}_{resolution}_bs{batch size}_{criterion}_{optimizer}_lr{lr}_{scheduler}_{epochs}e.yaml
```

- `{dataset}`: dataset name like `lane`, etc.
- `{method}`: method name like `resnet-18`, `efficientnet-b0`, etc.
- `{resolution}`: input resolution like `2400x1080`, `1800x1080`, etc.
- `{batch size}`: batch size during training, e.g. `2`, `4`.
- `{criterion}`: criterion name like `ce`, `focal`, etc.
- `{optimizer}`: optimizer name like `sgd`, `adam`, etc.
- `{lr}`: basic learning rate for training, e.g. `0.1`, `0.01`.
- `{scheduler}`: scheduler name like `exp`, `poly`, etc.
- `{epochs}`: epochs for training, e.g. `8`, `16`.

## <a name="train"></a> Train

```shell
$ export CUDA_VISIBLE_DEVICES=0,1
$ python train.py configs/lane_efficientnet-b0_1800x1080_bs4_ce_sgd_lr0.01_poly0.9_8e.yaml \
                  --comment 'use new backbone' \
                  --seed 1234 \
                  --configs MODEL.NAME resnet-152 \
                            DATASET.ROOT F:\Yao\Data\road \
                            DATALOADER.BATCH_SIZE 2
```

## <a name="submit"></a> Submit (After Training)

```shell
$ export CUDA_VISIBLE_DEVICES=0
$ cd runs/20220804-180000_lane_efficientnet-b0_1800x1080_bs4_ce_sgd_lr0.01_poly0.9_8e_use-new-backbone/
$ python ../../submit.py config.yaml last.pth
```