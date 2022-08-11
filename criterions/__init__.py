import torch.nn as nn
from torchtoolbox.nn import LabelSmoothingLoss

from configs import CFG
from .ohem import OHEMLoss


def build_criterion():
    if CFG.CRITERION.NAME == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif CFG.CRITERION.NAME == 'ohem-ce':
        criterion = OHEMLoss(nn.CrossEntropyLoss(reduction='none'))
    elif CFG.CRITERION.NAME == 'label-smoothing':
        criterion = LabelSmoothingLoss(CFG.DATASET.NUM_CLASSES, smoothing=CFG.CRITERION.SMOOTHING)
    # elif CFG.CRITERION.NAME == 'ml-sm':
    #     criterion = nn.MultiLabelSoftMarginLoss()
    else:
        raise NotImplementedError(f'invalid criterion: {CFG.CRITERION.NAME}')
    return criterion
