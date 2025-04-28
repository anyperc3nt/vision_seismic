import matplotlib.pyplot as plt
import numpy as np
import torch
from CFG import CFG
from matplotlib.animation import FuncAnimation
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm.auto import tqdm, trange


def train_model(train_loader, val_loader, model, criterion, optimizer, device):
    train_losses = []
    val_losses = []
    val_ssim_scores = []

    for epoch in tqdm(range(CFG.EPOCHS), desc="training", leave=True):
        if epoch == 0:
            print(
                f"{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}{'Val SSIM':<15}{'Log Train Loss':<15}{'Log Val Loss':<15}{'Log SSIM':<15}"
            )

        # Тренировочный цикл
        model.train()
        epoch_loss = 0
        SSIM_metric = StructuralSimilarityIndexMeasure().to(device)

        for images, targets in train_loader:
            images = images.to(device).to(dtype=torch.float32)
            targets = targets.to(device).to(dtype=torch.float32)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Валидация
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_score = 0
            for images, targets in val_loader:
                images = images.to(device).to(dtype=torch.float32)
                targets = targets.to(device).to(dtype=torch.float32)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                val_score += SSIM_metric(outputs, targets).item()

            val_loss /= len(val_loader)
            val_score /= len(val_loader)
            val_losses.append(val_loss)
            val_ssim_scores.append(val_score)

            log_epoch_loss = np.log(epoch_loss) if epoch_loss > 0 else float("-inf")  # Проверка на 0
            log_val_loss = np.log(val_loss) if val_loss > 0 else float("-inf")  # Проверка на 0
            log_val_score = np.log(val_score) if val_score > 0 else float("-inf")  # Проверка на 0

            print(
                f"{epoch+1:3}/{CFG.EPOCHS:<5}   {epoch_loss:<15.4f}{val_loss:<15.4f}{val_score:<15.4f}{log_epoch_loss:<15.4f}{log_val_loss:<15.4f}{log_val_score:<15.4f}"
            )

    return train_losses, val_losses, val_ssim_scores
