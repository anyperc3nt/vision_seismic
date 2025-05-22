import torch
import torch.nn as nn
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


class SimpleSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, preds, targets):
        return 1 - self.ssim(preds, targets)


class SSIM_MSE_Loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.ssim = SimpleSSIMLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # вес для SSIM

    def forward(self, preds, targets):
        return self.alpha * self.ssim(preds, targets) + (1 - self.alpha) * self.mse(preds, targets)
