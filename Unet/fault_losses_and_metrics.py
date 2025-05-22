import kornia  # для SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_optimizer.loss import TverskyLoss
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

class FaultIoU(BinaryJaccardIndex):
    """
    Считает IoU только на channel (как правило это fautls, 3 канал)
    """

    def __init__(self, channel=3, threshold=0.5):
        super().__init__(threshold=threshold, zero_division=1)
        self.channel = channel
        self.threshold = threshold

    def forward(self, preds, target):
        preds = preds[:, self.channel : self.channel + 1, :, :]
        target = target[:, self.channel : self.channel + 1, :, :]
        preds = (preds > self.threshold).float()
        target = (target > self.threshold).float()
        return super().forward(preds.reshape(-1), target.reshape(-1))


class SSIM3Channels(StructuralSimilarityIndexMeasure):
    """
    Считает SSIM только на channel (как правило это geomodel, первые 3 канала)
    """
    def __init__(self):
        super().__init__(data_range=1.0)

    def forward(self, preds, target):
        preds = preds[:, :3, :, :]
        target = target[:, :3, :, :]
        return super().forward(preds, target)


class CombinedMSETverskyLoss(nn.Module):
    """
    комбинированный лосс, 0-3 каналы - mse, 4 канал - фолт Tversky
    """

    def __init__(self, mse_weight=0.3, tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, pred, target):
        pred_mse = pred[:, :3, :, :]
        target_mse = target[:, :3, :, :]

        pred_tversky = pred[:, 3:4, :, :]
        target_tversky = target[:, 3:4, :, :]

        loss_mse = self.mse_loss(pred_mse, target_mse)

        # Внутри TverskyLoss заменяем view -> reshape
        pred_tversky = pred_tversky.reshape(-1)
        target_tversky = target_tversky.reshape(-1)
        loss_tversky = self.tversky_loss(pred_tversky, target_tversky)

        return loss_mse * self.mse_weight + loss_tversky

    def __repr__(self):
        return (f"CombinedMSETverskyLoss(mse_weight={self.mse_weight}, "
            f"tversky_alpha={self.tversky_alpha}, tversky_beta={self.tversky_beta})")


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        return kornia.losses.ssim_loss(pred, target, window_size=self.window_size, reduction="mean")


class CombinedMSE_SSIM_Loss(nn.Module):
    """
    комбинированный лосс, 0-3 каналы - mse, 4 канал - фолт SSIM
    """

    def __init__(self, mse_weight=0.3, fault_weight=0.7):
        super().__init__()
        self.mse_weight = mse_weight
        self.fault_weight = fault_weight
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target):
        pred_mse = pred[:, :3, :, :]
        target_mse = target[:, :3, :, :]

        pred_fault = pred[:, 3:4, :, :]
        target_fault = target[:, 3:4, :, :]

        loss_mse = self.mse_loss(pred_mse, target_mse)
        loss_fault = self.ssim_loss(pred_fault, target_fault)

        return loss_mse * self.mse_weight + loss_fault * self.fault_weight
