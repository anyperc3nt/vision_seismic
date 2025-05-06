import matplotlib.pyplot as plt
import torch
from data_preprocessing import make_seism_human_readable
from matplotlib.colors import LogNorm


def collect_predictions(model, dataset, indices, device):
    input_list = []
    target_list = []
    output_list = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, target = dataset[idx]
            image = image.unsqueeze(0).to(device, dtype=torch.float32)
            target = target.unsqueeze(0).to(device, dtype=torch.float32)

            output = model(image)

            input_list.append(image.squeeze(0).cpu())
            target_list.append(target.squeeze(0).cpu())
            output_list.append(output.squeeze(0).cpu())

    inputs = torch.stack(input_list)  # (N, C, H, W)
    targets = torch.stack(target_list)
    predictions = torch.stack(output_list)

    return inputs, targets, predictions


def standardize_img(img):
    """
    вспомогательная функция для приведения картинок к matplotlib-friendly формату
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        elif img.shape[0] == 1:
            img = img[0]
        else:
            raise ValueError(f"Неизвестный формат тензора: {img.shape}")

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    return img


def visualize(*arrays_with_flags, vmin=1e-4, vmax=1, title=None, dpi=80, use_log_norm=False):
    """
    arrays_with_flags: список кортежей вида (массив_тензоров, флаг_типа)
        флаг_типа ∈ {'seismogram', 'target', 'predict'}
        Каждый массив должен иметь форму (N, C, H, W) или (N, H, W, C)

        vmin=1e-4, vmax=1 - для LogNorm
    """

    n_rows = len(arrays_with_flags[0][0])
    n_cols = len(arrays_with_flags)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), dpi=dpi)
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for col, (array, flag) in enumerate(arrays_with_flags):
        for row in range(n_rows):
            img = standardize_img(array[row])

            if flag == "seismogram":
                img = make_seism_human_readable(img)
                axes[row, col].imshow(img, extent=[0, 500, 0, 250])
            elif flag in ["target", "predict"]:
                # Если флаг use_log_norm активирован, применяем LogNorm
                norm = LogNorm(vmin=vmin, vmax=vmax) if use_log_norm else None
                axes[row, col].imshow(img[:, :, 0], extent=[0, 500, 0, 250], norm=norm)
            else:
                raise ValueError(f"Неизвестный флаг: {flag}")

            if row == 0:
                axes[row, col].set_title(
                    {
                        "seismogram": "Seismogram (3ch -> RGB)",
                        "target": "Geomodel: Ground Truth",
                        "predict": "Geomodel: Predict",
                    }[flag]
                )
            axes[row, col].axis("off")

    if title:
        plt.subplots_adjust(top=0.85)
        fig.suptitle(title, fontsize=16, x=0.5, y=0.95)

    plt.tight_layout()
    plt.show()
