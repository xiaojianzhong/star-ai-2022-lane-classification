import torch.optim as optim

from configs import CFG


def build_scheduler(optimizer):
    assert not (CFG.SCHEDULER.BY_EPOCH and CFG.SCHEDULER.BY_ITERATION)

    if CFG.SCHEDULER.NAME == '':
        # scheduler is allowed to be None, which means the learning rate wouldn't be changed during training
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=lambda _: 1)
    elif CFG.SCHEDULER.NAME == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=CFG.SCHEDULER.GAMMA)
    elif CFG.SCHEDULER.NAME == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=lambda iteration: (1 - iteration / CFG.ITERATIONS) ** CFG.SCHEDULER.GAMMA)
    else:
        raise NotImplementedError('invalid scheduler: {}'.format(CFG.SCHEDULER.NAME))
    return scheduler
