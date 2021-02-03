# Borrowed from https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
import torch
import torch.nn.functional as F
import math


class GenericRandomResizedCrop():
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(x, scale, ratio):
        width, height = x.shape[1:]
        area = height * width

        for _ in range(100):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, x):
        i, j, h, w = self.get_params(x, self.scale, self.ratio)
        x = x[:, j:j+w, i:i+h]
        return F.interpolate(x.unsqueeze(0), size=self.size, mode='bicubic', align_corners=True).squeeze(0)
