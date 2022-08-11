from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.transforms import *

from configs import CFG
from .lane import StarAI2022LaneDataset
from .transforms import AdaptiveCrop, AdjustColor


def build_transform(split):
    transforms = []
    transforms += [
        AdaptiveCrop(CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.ORIGIN_HEIGHTS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.ORIGIN_WIDTHS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.TOPS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.LEFTS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.HEIGHTS,
                     CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.WIDTHS),
        Resize((CFG.DATASET.TRANSFORM.RESIZE.HEIGHT,
                CFG.DATASET.TRANSFORM.RESIZE.WIDTH)),
    ]
    if split == 'train':
        transforms += [
            RandomHorizontalFlip(p=CFG.DATASET.TRANSFORM.RANDOM_HORIZONTAL_FLIP.PROBABILITY),
            RandomRotation(CFG.DATASET.TRANSFORM.RANDOM_ROTATION.DEGREE, fill=(232, 242, 221)),
        ]
    transforms += [
        AdjustColor(CFG.DATASET.TRANSFORM.BRIGHTNESS,
                    CFG.DATASET.TRANSFORM.CONTRAST),
        ToTensor(),
        Normalize(mean=CFG.DATASET.TRANSFORM.NORMALIZE.MEAN,
                  std=CFG.DATASET.TRANSFORM.NORMALIZE.STD),
    ]
    return Compose(transforms)


def build_dataset(split):
    if CFG.DATASET.NAME == 'lane':
        dataset = StarAI2022LaneDataset(CFG.DATASET.ROOT,
                                        split,
                                        transform=build_transform(split))
    else:
        raise NotImplementedError(f'invalid dataset: {CFG.DATASET.NAME}')
    return dataset


def build_dataloader(dataset, split):
    if split == 'train':
        return DataLoader(dataset,
                          batch_size=CFG.DATALOADER.BATCH_SIZE,
                          # shuffle=True,
                          sampler=ImbalancedDatasetSampler(dataset),  # use imbalanced dataset sampler
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=True)
    elif split == 'test':
        return DataLoader(dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False)
    else:
        raise NotImplementedError(f'invalid split: {split}')
