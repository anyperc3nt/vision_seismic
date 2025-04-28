import numpy as np
from CFG import CFG


def normalize_geomodel(geomodel_3ch):
    """
    приводит геомодели к [0,1] в соответствие с заданными в конфиге физическими параметрами
    """
    geomodel_3ch[..., 0] = (geomodel_3ch[..., 0] - CFG.rho_range[0]) / (CFG.rho_range[1] - CFG.rho_range[0])
    geomodel_3ch[..., 1] = (geomodel_3ch[..., 1] - CFG.vp_range[0]) / (CFG.vp_range[1] - CFG.vp_range[0])
    geomodel_3ch[..., 2] = (geomodel_3ch[..., 2] - CFG.vs_range[0]) / (CFG.vs_range[1] - CFG.vs_range[0])

    return geomodel_3ch


def normalize_seism(seism_3ch):
    # Находим минимальное и максимальное значение по всем каналам
    seism_min = np.min(seism_3ch)
    seism_max = np.max(seism_3ch)

    # Нормализуем все каналы относительно глобального минимума и максимума
    normalized_seism = (seism_3ch - seism_min) / (seism_max - seism_min)

    return normalized_seism


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

    for i in range(seism_3ch.shape[-1]):  # Перебираем каналы
        channel = seism_3ch[..., i]  # Берем один канал
        # Находим медиану канала
        median_value = np.median(channel)
        # Центрируем канал относительно медианы
        normalized_channel = channel - median_value
        normalized[..., i] = normalized_channel  # Записываем обратно в массив

    return normalized


def make_seism_human_readable(seism_3ch):
    # Находим минимальное и максимальное значение по всем каналам
    seism_min = np.min(seism_3ch)
    seism_max = np.max(seism_3ch)

    # Нормализуем все каналы относительно глобального минимума и максимума
    seism_3ch = (seism_3ch - seism_min) / (seism_max - seism_min)

    seism_3ch = normalize_to_median(seism_3ch)
    rows = np.arange(seism_3ch.shape[0])
    seism_3ch = seism_3ch * ((100 + rows) ** 1.7)[:, np.newaxis, np.newaxis]
    seism_3ch = normalize_seism(seism_3ch)
    seism_3ch = contrast_stretch(seism_3ch)  # постарались пофиксить распределения

    return seism_3ch
