import os

import numpy as np
from constants import Constants
from skimage.transform import resize
from tqdm.auto import tqdm, trange


class DataIO:
    """
    работа с парсингом txt или bin файлов
    загружает данные из датасета и возвращает в формате HWC
    """

    @staticmethod
    def load_seism_raw(paths, index):
        seism_channels = []
        for channel_path in paths[index]:  # C
            channel_data = np.loadtxt(channel_path, dtype="f")[:, 1:]
            # [:,1:] - это важно, мы выкидываем 1 столбик в котором время
            seism_channels.append(channel_data)
        # HWC
        return np.stack(seism_channels, axis=-1)

    @staticmethod
    def load_geomodel_raw(paths, index):
        geo_channels = []
        for channel_path in paths[index]:  # C
            channel_data = np.fromfile(channel_path, dtype="f")
            channel_data = channel_data.reshape(Constants.model_y_size, Constants.model_x_size)[::-1, :]
            # [::-1, :] - флипаем модельки
            geo_channels.append(channel_data)
        # HWC
        return np.stack(geo_channels, axis=-1)

    @staticmethod
    def load_fault_raw(paths, index):
        fault = np.fromfile((paths[index]), dtype="f")
        fault = fault.reshape(Constants.model_y_size, Constants.model_x_size, 1)[::-1, :, :]
        # [::-1, :] - флипаем модельки
        # HW1
        return fault


def normalize_geomodel(geomodel_3ch):
    """
    приводит геомодели к [0,1] в соответствие с заданными в constants физическими параметрами
    """
    geomodel_3ch[..., 0] = (geomodel_3ch[..., 0] - Constants.rho_range[0]) / (
        Constants.rho_range[1] - Constants.rho_range[0]
    )
    geomodel_3ch[..., 1] = (geomodel_3ch[..., 1] - Constants.vp_range[0]) / (
        Constants.vp_range[1] - Constants.vp_range[0]
    )
    geomodel_3ch[..., 2] = (geomodel_3ch[..., 2] - Constants.vs_range[0]) / (
        Constants.vs_range[1] - Constants.vs_range[0]
    )

    return geomodel_3ch


def normalize_seism(seism_3ch):
    """
    тут нормализация относительно минимума и максимума -
    по сути, нормализуемся на источник
    """
    # Находим минимальное и максимальное значение по всем каналам
    seism_min = np.min(seism_3ch)
    seism_max = np.max(seism_3ch)

    # Нормализуем все каналы относительно глобального минимума и максимума
    normalized_seism = (seism_3ch - seism_min) / (seism_max - seism_min)

    return normalized_seism


def resize_to_stretch(x, shape):
    # order=0 — nearest neighbor, просто растягивает без сглаживания
    # order=1 — линейная интерполяция, плавное растяжение
    return resize(x, shape, order=0, mode="edge", preserve_range=True).astype(np.float32)


class NpyBuilder:
    """
    промежуточный "датасет" задача которого сохранить распарсенные
    и предобработанные файлики в быстром для чтения формате
    """

    def __init__(
        self,
        BASE_PATH,
        seism_paths,
        geomodel_paths,
        fault_paths=None,
        force=False,
    ):
        self.BASE_PATH = BASE_PATH
        self.seism_paths = seism_paths
        self.geomodel_paths = geomodel_paths
        self.fault_paths = fault_paths

        os.makedirs(f"{BASE_PATH}/npy", exist_ok=True)
        # сколько десятичных знаков в нумеровке семплов (если 5, 1 преобразуется в 00001 для сортировки)
        digits = len(str(len(self)))
        # Проверяем, нужно ли пересоздавать npy
        npy_exists = os.path.exists(f"{BASE_PATH}/npy/vp_vs_geo_{len(seism_paths)-1:0{digits}}.npy")

        if (not npy_exists) or force:
            print("Building .npy dataset for fast loading... (it'll be done only once)")
            os.makedirs(f"{BASE_PATH}/npy", exist_ok=True)
            file_list_path = f"{BASE_PATH}/npy/file_list.txt"

            with open(file_list_path, "w") as f:
                for i in tqdm(range(len(self))):
                    sample = self[i]
                    if sample is None:
                        continue

                    seism_vp, seism_vs, geomodel = sample
                    combined = np.concatenate([seism_vp, seism_vs, geomodel], axis=-1)
                    np.save(f"{BASE_PATH}/npy/vp_vs_geo_{i:0{digits}}.npy", combined)
                    f.write(f"{BASE_PATH}/npy/vp_vs_geo_{i:0{digits}}.npy\n")

    def __len__(self):
        return len(self.seism_paths)

    def __getitem__(self, index):

        seism = DataIO.load_seism_raw(self.seism_paths, index)
        seism = normalize_seism(seism)

        if np.isinf(seism).any() or np.isnan(seism).any():
            return None

        geomodel = DataIO.load_geomodel_raw(self.geomodel_paths, index)
        geomodel = normalize_geomodel(geomodel)

        if self.fault_paths is not None:
            fault = DataIO.load_fault_raw(self.fault_paths, index)
            geomodel = np.concatenate([geomodel, fault], axis=-1)

        seism_vp = seism[:, 1::2, :]  # 500 624 3 -> 500 312 3
        seism_vs = seism[:, 0::2, :]

        seism_vp = resize(seism_vp, Constants.sample_shape, order=0, mode="edge", preserve_range=True)
        seism_vs = resize(seism_vs, Constants.sample_shape, order=0, mode="edge", preserve_range=True)
        geomodel = resize(geomodel, Constants.sample_shape, order=0, mode="edge", preserve_range=True)

        return seism_vp, seism_vs, geomodel
