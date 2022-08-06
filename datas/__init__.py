from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.transforms import *

from configs import CFG
from .lane import StarAI2022LaneDataset
from .transforms import AdaptiveCrop


def build_transform():
    return Compose([
        AdaptiveCrop(CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.ORIGIN_HEIGHTS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.ORIGIN_WIDTHS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.TOPS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.LEFTS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.HEIGHTS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.WIDTHS),
        Resize((CFG.DATASET.TRANSFORM.RESIZE.HEIGHT, CFG.DATASET.TRANSFORM.RESIZE.WIDTH)),
        # RandomResizedCrop((CFG.DATASET.TRANSFORM.RESIZE.HEIGHT, CFG.DATASET.TRANSFORM.RESIZE.WIDTH)),
        # RandomHorizontalFlip(),
        # RandomRotation(90),
        ToTensor(),
        Normalize(mean=CFG.DATASET.TRANSFORM.NORMALIZE.MEAN,
                  std=CFG.DATASET.TRANSFORM.NORMALIZE.STD),
    ])


def build_dataset():
    if CFG.DATASET.NAME == 'lane':
        dataset = StarAI2022LaneDataset(CFG.DATASET.ROOT,
                                        transform=build_transform())
    else:
        raise NotImplementedError(f'invalid dataset: {CFG.DATASET.NAME}')
    return dataset


def build_dataloader(dataset):
    return DataLoader(dataset,
                      batch_size=CFG.DATALOADER.BATCH_SIZE,
                      # shuffle=True,
                      sampler=ImbalancedDatasetSampler(dataset),  # use imbalanced dataset sampler
                      num_workers=CFG.DATALOADER.NUM_WORKERS,
                      pin_memory=True,
                      drop_last=True)
