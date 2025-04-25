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
