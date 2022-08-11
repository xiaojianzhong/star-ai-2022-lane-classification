import torch.nn as nn
from torchvision.transforms.functional import crop, adjust_brightness, adjust_contrast


class AdaptiveCrop(nn.Module):
    def __init__(self, origin_heights, origin_widths, tops, lefts, heights, widths):
        super(AdaptiveCrop, self).__init__()

        assert len(origin_heights) == len(origin_widths) == len(tops) == len(lefts) == len(heights) == len(widths)
        self.num_types = len(origin_heights)

        self.origin_heights = origin_heights
        self.origin_widths = origin_widths
        self.tops = tops
        self.lefts = lefts
        self.heights = heights
        self.widths = widths
        self.topleft = (300, 0)
        self.size = (1800, 1080)

    def forward(self, img):
        if self.num_types == 0:
            return img
        else:
            for i in range(self.num_types):
                if img.size == (self.origin_widths[i], self.origin_heights[i]):
                    return crop(img, self.tops[i], self.lefts[i], self.heights[i], self.widths[i])
            raise AssertionError(f'img.size == {img.size}')

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AdjustColor(nn.Module):
    def __init__(self, brightness_factor, contrast_factor):
        super(AdjustColor, self).__init__()

        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def forward(self, img):
        img = adjust_brightness(img, self.brightness_factor)
        img = adjust_contrast(img, self.contrast_factor)
        return img
