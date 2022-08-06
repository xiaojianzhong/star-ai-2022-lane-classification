import torch.nn as nn


class OHEMLoss(nn.Module):
    def __init__(self, criterion, ratio_max_kept=0.8, size_average=True):
        super(OHEMLoss, self).__init__()
        self.criterion = criterion
        self.ratio_max_kept = ratio_max_kept
        self.size_average = size_average

    def forward(self, input, target):
        n = input.shape[0]

        batch_losses = self.criterion(input, target)
        topk_losses, _ = batch_losses.topk(k=int(n * self.ratio_max_kept))

        if self.size_average:
            loss = topk_losses.mean()
        else:
            loss = topk_losses.sum()
        return loss
