import torch.optim as optim

from configs import CFG


def build_optimizer(params):
    if CFG.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(params,
                              lr=CFG.OPTIMIZER.LR,
                              momentum=CFG.OPTIMIZER.MOMENTUM,
                              weight_decay=CFG.OPTIMIZER.WEIGHT_DECAY)
    elif CFG.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(params,
                               lr=CFG.OPTIMIZER.LR,
                               weight_decay=CFG.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError(f'invalid optimizer: {CFG.OPTIMIZER.NAME}')
    return optimizer

