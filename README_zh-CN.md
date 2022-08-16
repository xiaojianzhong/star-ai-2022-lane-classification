# 2022 全球校园 AI 算法精英赛道 - 车道渲染数据智能质检

阅读其他语言的版本：[English](README.md) | 简体中文

## 目录

- [环境要求](#prerequisites)
- [配置文件名格式](#configuration-name-format)
- [准备工作](#preparation)
- [训练](#train)
- [（训练完成后）提交](#submit)
- [（可选）恢复训练](#resume-training)

## <a name="prerequisites"></a> 环境要求

- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pillow](https://python-pillow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [torch](https://pytorch.org/)
- [torchinfo](https://github.com/tyleryep/torchinfo)
- [torchsampler](https://github.com/ufoym/imbalanced-dataset-sampler)
- [torchtoolbox](https://github.com/PistonY/torch-toolbox)
- [torchvision](https://pytorch.org/vision/)
- [tqdm](https://github.com/tqdm/tqdm)
- [yacs](https://github.com/rbgirshick/yacs)

所有这些 Python 第三方包都可以简单地使用 `pip` 进行安装：

```shell
$ pip install -r requirements.txt
```

## <a name="configuration-name-format"></a> 配置文件名格式

```
{dataset}_{method}_{resolution}_bs{batch size}_{criterion}_{optimizer}_lr{lr}_{scheduler}_{epochs}e.yaml
```

- `{dataset}`: 数据集名称，如 `lane`
- `{method}`: 方法名称，如 `resnet-18`，`efficientnet-b0`
- `{resolution}`: 输入分辨率，如 `2400x1080`，`1800x1080`
- `{batch size}`: 训练过程中的批次大小，如 `2`，`4`
- `{criterion}`: 损失函数名称，如 `ce`，`focal`
- `{optimizer}`: 优化器名称，如 `sgd`，`adam`
- `{lr}`: 用于训练的基础学习率，如 `0.1`，`0.01`
- `{scheduler}`: 学习率调整器，如 `exp`，`poly`
- `{epochs}`: 训练轮数，如 `8`，`16`

## <a name="preparation"></a> 准备工作

假定你的数据集存放在 `DATASET_PATH` 路径下，

```shell
$ mkdir DATASET_PATH/train_label/labeled_data/
$ mv DATASET_PATH/train_label/train_label.csv DATASET_PATH/train_label/labeled_data/
$ mkdir DATASET_PATH/train_label/unlabeled_data/
$ python dir2csv.py DATASET_PATH/train_label/unlabeled_data/ DATASET_PATH/train_label/unlabeled_data/train_label.csv
$ mkdir DATASET_PATH/test_label/
$ python dir2csv.py DATASET_PATH/test_label/ DATASET_PATH/test_label/test_label.csv
```

## <a name="train"></a> 训练

```shell
$ export CUDA_VISIBLE_DEVICES=0,1
$ python train.py configs/lane_efficientnet-b0_1800x1080_bs4_ce_sgd_lr0.01_poly0.9_8e.yaml \
                  --comment 'use new backbone' \
                  --seed 1234 \
                  --configs MODEL.NAME resnet-152 \
                            DATASET.ROOT F:\Yao\Data\road \
                            DATALOADER.BATCH_SIZE 2
```

## <a name="submit"></a> （训练完成后）提交

```shell
$ export CUDA_VISIBLE_DEVICES=0,1
$ cd runs/20220804-180000_lane_efficientnet-b0_1800x1080_bs4_ce_sgd_lr0.01_poly0.9_8e_use-new-backbone/
$ python ../../submit.py config.yaml last.pth --configs DATALOADER.BATCH_SIZE 64
```

为提高推理速度，你可以通过指定 `--configs` 参数来增加 `DATALOADER.BATCH_SIZE`，如 `--configs DATALOADER.BATCH_SIZE 64`。

## <a name="resume-training"></a> （可选）恢复训练

```shell
$ python train.py configs/lane_efficientnet-b0_1800x1080_bs4_ce_sgd_lr0.01_poly0.9_8e.yaml \
                  --checkpoint runs/20220804-180000_lane_efficientnet-b0_1800x1080_bs4_ce_sgd_lr0.01_poly0.9_8e_use-new-backbone/last.pth \
                  --comment 'resume from previous checkpoint'
2022-08-04 18:00:00,000: load checkpoint runs/20220804-180000_lane_efficientnet-b0_1800x1080_bs4_ce_sgd_lr0.01_poly0.9_8e_use-new-backbone/last.pth with f1=0.8500
......
```
