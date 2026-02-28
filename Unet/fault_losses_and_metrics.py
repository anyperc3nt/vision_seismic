import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_optimizer.loss import TverskyLoss as BaseTverskyLoss
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score 
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


class LossCombinator(nn.Module):
    """
    используется при создании составных лоссов для мультизадачи

    losses: список экземпляров лоссов
    channels: список элементов вида [начальный канал, конечный канал],
        к которым этот лосс будет применяться
    weights: веса для каждого лосса
    flatten_channels: необходимость "выпремлять" батч для совместимости с лоссом
    """

    def __init__(self, losses: list, channels: list, weights=None, flatten_channels=None):
        super().__init__()
        assert len(losses) == len(channels)
        if weights is None:
            weights = [1.0] * len(losses)
        assert len(weights) == len(losses)
        self.losses = nn.ModuleList(losses)
        self.channels = channels
        self.weights = weights

        # флаг, надо ли сплющивать для каждого лосса, по умолчанию True для бинарных (одноканальных)
        if flatten_channels is None:
            flatten_channels = []
            for start, end in channels:
                flatten_channels.append((end - start) == 1)
        assert len(flatten_channels) == len(losses)
        self.flatten_channels = flatten_channels

    def forward(self, pred, target):
        total_loss = 0
        for loss_fn, (start, end), weight, flatten in zip(
            self.losses, self.channels, self.weights, self.flatten_channels
        ):
            pred_part = pred[:, start:end, :, :]
            target_part = target[:, start:end, :, :]

            if flatten:
                pred_part = pred_part.reshape(-1)
                target_part = target_part.reshape(-1)

            total_loss += weight * loss_fn(pred_part, target_part)
        return total_loss

    def __repr__(self):
        lines = ["LossCombinator composed of:"]
        for loss_fn, (start, end), weight in zip(self.losses, self.channels, self.weights):
            lines.append(f"  weight={weight}: {loss_fn.__repr__()} (channels {start}:{end})")
        return "\n".join(lines)

    def __str__(self):
        return "_".join([loss.__class__.__name__[:4] for loss in self.losses])


class TverskyLoss(BaseTverskyLoss):
    def __init__(self, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits

    def forward(self, input, target):
        if self.from_logits:
            input = torch.sigmoid(input)
        return super().forward(input, target)

    def __repr__(self):
        return (
            f"TverskyLoss(alpha={self.alpha}, beta={self.beta}, "
            f"smooth={self.smooth}, from_logits={self.from_logits})"
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma})"

class FaultIoU(BinaryJaccardIndex):
    """
    Считает IoU по заданному каналу (например, faults, канал 3).
    Может принимать логиты, если with_logits=True.
    """

    def __init__(self, channel, threshold=0.5, with_logits=True):
        super().__init__(threshold=threshold, zero_division=1)
        self.channel = channel
        self.threshold = threshold
        self.with_logits = with_logits

    def forward(self, preds, target):
        preds = preds[:, self.channel : self.channel + 1, :, :]
        target = target[:, self.channel : self.channel + 1, :, :]
        if self.with_logits:
            preds = torch.sigmoid(preds)
        # preds = (preds > self.threshold).float()
        target = (target > 0.5).int()
        return super().forward(preds.reshape(-1), target.reshape(-1))

class SSIM3Channels(StructuralSimilarityIndexMeasure):
    """
    Считает SSIM только на [:3] каналы
    """

    def __init__(self):
        super().__init__(data_range=1.0)

    def forward(self, preds, target):
        preds = preds[:, :3, :, :]
        target = target[:, :3, :, :]
        return super().forward(preds, target)
