import matplotlib.pyplot as plt
import numpy as np
import torch
from visualization.visual_preprocessing import make_seism_human_readable


def standardize_img(img):
    """
    вспомогательная функция для приведения картинок к HWC формату для matplotlib отображения
    torch.Tensor -> np.ndarray
    3HW -> HW3
    HW3 -> HW3
    1HW -> HW
    HW1 -> HW
    6HW -> HW6
    HW6 -> HW6
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.shape[0] in [3, 4, 6]:
        img = img.transpose(1, 2, 0)
    elif img.shape[0] == 1:
        img = img[0]
    else:
        raise ValueError(f"Неизвестный формат тензора: {img.shape}")

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def make_seism_img(seism):
    """
    seism: np.ndarray[H, W, C]
    Визуализирует сейсмограмму: 3 или 6 каналов.
    При 6 каналах отображает два изображения рядом: 0:3 и 3:6.
    """
    if seism.shape[2] == 3:
        return make_seism_human_readable(seism)
    elif seism.shape[2] == 6:
        img_left = make_seism_human_readable(seism[:, :, :3])
        img_right = make_seism_human_readable(seism[:, :, 3:])
        gap = np.ones((img_left.shape[0], 5, 3))
        return np.hstack([img_left, gap, img_right])
    else:
        raise ValueError(f"Unexpected number of channels: {seism.shape[2]}")


def make_geo_img(geo):
    """
    geo: np.ndarray[H, W, C]
    Визуализирует геомодель: 3 или 4 канал.
    при 3 каналах отображает 0(rho)
    при 4 каналах отображает два изображения рядом: 0(rho) и 3 (фолты).
    """
    if len(geo.shape) == 2:
        return geo
    elif geo.shape[2] == 3:
        return geo[:, :, 0]
    elif geo.shape[2] == 4:
        img_left = geo[:, :, 0]
        img_right = geo[:, :, 3]
        gap = np.ones((img_left.shape[0], 5))
        return np.hstack([img_left, gap, img_right])
    else:
        raise ValueError(f"Unexpected number of channels: {geo.shape[2]}")


def make_samples_fig(seisms=None, geo_predicts=None, geo_targets=None):
    """
    Собирает одну общую фигуру с тремя столбцами (опиональными):
    сейсмограммы, предсказанные геомодели, таргеты.
    Каждое изображение обрабатывается через standardize_img, затем визуализируется.
    """
    imgs = []
    titles = []

    if seisms is not None:
        imgs.append([make_seism_img(standardize_img(img)) for img in seisms])
        titles.append("Seism")
    if geo_predicts is not None:
        imgs.append([make_geo_img(standardize_img(img)) for img in geo_predicts])
        titles.append("Predict")
    if geo_targets is not None:
        imgs.append([make_geo_img(standardize_img(img)) for img in geo_targets])
        titles.append("Target")

    cols = len(imgs)
    rows = len(imgs[0]) if imgs else 0

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axs = [[axs]]
    elif rows == 1:
        axs = [axs]
    elif cols == 1:
        axs = [[ax] for ax in axs]

    for row in range(rows):
        for col in range(cols):
            axs[row][col].imshow(imgs[col][row])
            axs[row][col].axis("off")

    for col, title in enumerate(titles):
        axs[0][col].set_title(title, fontsize=14)

    plt.tight_layout()
    return fig


def make_losses_and_metrics_fig(
    losses: list,
    loss_labels: list,
    metrics=None,
    metrics_labels=None,
    title=None,
    xlabel="Epoch",
    ylabel="Value",
    logscale=False,
):
    if metrics is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(6, 4))

    colormap = plt.cm.viridis

    for i, (array, label) in enumerate(zip(losses, loss_labels)):
        color = colormap(i / len(losses))
        ax1.plot(range(1, len(array) + 1), array, label=label, color=color, linestyle="-")

    ax1.set_ylabel(ylabel)
    if title:
        ax1.set_title(title)
    if logscale:
        ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True)

    if metrics is not None:
        for i, (array, label) in enumerate(zip(metrics, metrics_labels)):
            color = colormap(i / len(metrics))
            ax2.plot(range(1, len(array) + 1), array, label=label, color=color, linestyle="--")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Metric")
        ax2.legend()
        ax2.grid(True)
    else:
        ax1.set_xlabel(xlabel)

    plt.tight_layout()
    return fig
