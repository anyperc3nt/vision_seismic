import numpy as np
from data_pipeline.data_preprocessing import normalize_seism

"""
раздел, нужный для отображения сейсмограмм хорошими видимыми картинками
"""


def contrast_stretch(seism_3ch, lower_percentile=3, upper_percentile=97):
    # Нахождение минимумов и максимумов по перцентилям
    min_val = np.percentile(seism_3ch, lower_percentile, axis=(0, 1), keepdims=True)
    max_val = np.percentile(seism_3ch, upper_percentile, axis=(0, 1), keepdims=True)

    # Применение растяжки контраста
    seism_3ch_stretched = np.clip(seism_3ch, min_val, max_val)
    seism_3ch_stretched = (seism_3ch_stretched - min_val) / (max_val - min_val)

    return seism_3ch_stretched


def normalize_to_median(seism_3ch):
    normalized = np.empty_like(seism_3ch, dtype=np.float32)

    for i in range(seism_3ch.shape[-1]):
        channel = seism_3ch[..., i]
        median_value = np.median(channel)
        normalized_channel = channel - median_value
        normalized[..., i] = normalized_channel

    return normalized


def make_seism_human_readable(seism_3ch):
    seism_min = np.min(seism_3ch)
    seism_max = np.max(seism_3ch)

    seism_3ch = (seism_3ch - seism_min) / (seism_max - seism_min)

    seism_3ch = normalize_to_median(seism_3ch)
    rows = np.arange(seism_3ch.shape[0])
    seism_3ch = seism_3ch * ((100 + rows) ** 1.7)[:, np.newaxis, np.newaxis]
    seism_3ch = normalize_seism(seism_3ch)
    seism_3ch = contrast_stretch(seism_3ch)

    return seism_3ch
