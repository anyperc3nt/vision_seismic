import torch
from tqdm.auto import tqdm, trange
from CFG import CFG


def train_model(train_loader, val_loader, model, criterion, optimizer, device):
    for epoch in range(CFG.EPOCHS):
        model.train()
        epoch_loss = 0

        for images, targets in tqdm(train_loader, desc="training", leave=False):
            images = images.to(device).to(dtype=torch.float32)
            targets = targets.to(device).to(dtype=torch.float32)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{CFG.EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

        # Валидация
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, targets in tqdm(val_loader, desc="validation", leave=False):
                images = images.to(device).to(dtype=torch.float32)
                targets = targets.to(device).to(dtype=torch.float32)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
