import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from CFG import CFG
from data_preprocessing import normalize_geomodel, normalize_seism
from torch.utils.data import DataLoader, Dataset


def load_seism(seism_paths, index):

    seism_0 = np.loadtxt((seism_paths[index][0]), dtype="f")[:, 1:]
    seism_1 = np.loadtxt((seism_paths[index][1]), dtype="f")[:, 1:]
    seism_2 = np.loadtxt((seism_paths[index][2]), dtype="f")[:, 1:]
    seism_3ch = np.stack((seism_0, seism_1, seism_2), axis=-1)

    return seism_3ch


def load_seism_from_npy(seism_paths, index):

    original_path = seism_paths[index][0]
    parent_dir = os.path.dirname(os.path.dirname(original_path))
    filename = f"seism_3ch.npy"
    save_path = os.path.join(parent_dir, filename)

    seism_3ch = np.load(save_path)

    return seism_3ch


def bake_seisms_to_npy(seism_paths, force=False):
    """
    Txt-файлы долго парсятся.
    Функция загружает данные сейсмограмм из Txt, предобрабатывает их, сохраняет в быстро загружающиеся npy файлы
    Вызывается 1 раз при инициализации датасета

    Причем, эта работа проделывается лишь однократно для 1 датасета,
    потому что при повторном вызове (если не стоит force) она ее проделывать не будет
    """
    # проверяем, была ли выполнена эта работа до нас
    parent_dir = os.path.dirname(os.path.dirname(seism_paths[-1][0]))
    filename = f"seism_3ch.npy"
    save_path = os.path.join(parent_dir, filename)
    already_done = os.path.exists(save_path)

    if (not already_done) or (force):
        print("предобрабатываем и сохраняем сейсмограммы")
        for index in range(len(seism_paths)):
            seism_3ch = load_seism(seism_paths, index)
            seism_3ch = normalize_seism(seism_3ch)

            parent_dir = os.path.dirname(os.path.dirname(seism_paths[index][0]))
            filename = f"seism_3ch.npy"
            save_path = os.path.join(parent_dir, filename)

            np.save(save_path, seism_3ch)


def load_geomodel(model_paths, index):

    model_0 = np.fromfile((model_paths[index][0]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)[::-1, :]
    model_1 = np.fromfile((model_paths[index][1]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)[::-1, :]
    model_2 = np.fromfile((model_paths[index][2]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)[::-1, :]
    geomodel_3ch = np.stack((model_0, model_1, model_2), axis=-1)

    return geomodel_3ch


def load_geomodel_from_npy(model_paths, index):

    original_path = model_paths[index][0]
    dir = os.path.dirname(original_path)
    filename = f"geomodel_3ch.npy"
    save_path = os.path.join(dir, filename)

    geomodel_3ch = np.load(save_path)

    return geomodel_3ch


def bake_geomodels_to_npy(geomodel_paths, force=False):
    """
    Аналогично bake_seisms_to_npy
    """
    # проверяем, была ли выполнена эта работа до нас
    dir = os.path.dirname(geomodel_paths[-1][0])
    filename = f"geomodel_3ch.npy"
    save_path = os.path.join(dir, filename)
    already_done = os.path.exists(save_path)

    if (not already_done) or (force):
        print("предобрабатываем и сохраняем геомодели")
        for index in range(len(geomodel_paths)):
            geomodel_3ch = load_geomodel(geomodel_paths, index)
            geomodel_3ch = normalize_geomodel(geomodel_3ch)

            original_path = geomodel_paths[index][0]
            dir = os.path.dirname(original_path)
            filename = f"geomodel_3ch.npy"
            save_path = os.path.join(dir, filename)

            np.save(save_path, geomodel_3ch)


def load_fault(fault_paths, index):

    fault = np.fromfile((fault_paths[index]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)[::-1, :]

    return fault


base_transform = A.Compose(
    [
        A.Resize(
            height=CFG.seism_y_size, width=CFG.seism_x_size // CFG.CHANNEL_DELIMITER, interpolation=cv2.INTER_NEAREST
        ),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ],
    is_check_shapes=False,
    additional_targets={
        "seism1": "image",
        "seism2": "image",
    },
)


class Seism_Dataset(Dataset):
    def __init__(
        self,
        seism_paths,
        geomodel_paths,
        channel_tag=CFG.CHANNEL_TAG,
        transform=base_transform,
        force=False,
    ):
        """
        force - флаг, отвечающий за форсирование загрузки сейсмограмм из txt и геомоделей из бинарников на этапе инициализации,
        для того чтобы перезаписать npy файлики

        channel_type: - датасет возвращает сейсмограммы разных каналов в зависимости от выбранного подхода (см. CFG)
        """
        self.channel_tag = channel_tag
        self.seism_paths = seism_paths
        self.geomodel_paths = geomodel_paths
        self.transform = transform

        bake_seisms_to_npy(self.seism_paths, force=force)
        bake_geomodels_to_npy(self.geomodel_paths, force=force)

        seism, geomodel = self.__getitem__(0)
        print(f"seism shape: {seism.shape}, geomodel shape: {geomodel.shape}")

    def __len__(self):
        return len(self.seism_paths)

    def __getitem__(self, index):

        seism_3ch = load_seism_from_npy(self.seism_paths, index)
        geomodel_3ch = load_geomodel_from_npy(self.geomodel_paths, index)

        if self.channel_tag == "3ch":
            augmented = self.transform(image=geomodel_3ch, seism1=seism_3ch)
            geomodel = augmented["image"]
            seism = augmented["seism1"]

        elif self.channel_tag == "6ch":
            augmented = self.transform(image=geomodel_3ch, seism1=seism_3ch[:, 0::2], seism2=seism_3ch[:, 1::2])
            geomodel = augmented["image"]
            seism_vp = augmented["seism1"]  # пока что vp это четные, на самом деле хз какие
            seism_vs = augmented["seism2"]
            seism = torch.cat([seism_vp, seism_vs], dim=0)

        elif self.channel_tag == "vp":
            augmented = self.transform(image=geomodel_3ch, seism1=seism_3ch[:, 0::2], seism2=seism_3ch[:, 1::2])
            geomodel = augmented["image"]
            seism = augmented["seism1"]  # пока что vp это четные, на самом деле хз какие

        elif self.channel_tag == "vs":
            augmented = self.transform(image=geomodel_3ch, seism1=seism_3ch[:, 0::2], seism2=seism_3ch[:, 1::2])
            geomodel = augmented["image"]
            seism = augmented["seism2"]  # пока что vp это четные, на самом деле хз какие

        return seism, geomodel


class Seism_3ch_Dataset(Dataset):
    """
    старый класс, пусть пока останется
    """

    def __init__(self, seism_paths, geomodel_paths, transform=base_transform, force=False):
        self.seism_paths = seism_paths
        self.geomodel_paths = geomodel_paths
        self.transform = transform

        bake_seisms_to_npy(self.seism_paths, force=force)
        bake_geomodels_to_npy(self.geomodel_paths, force=force)

    def __len__(self):
        return len(self.seism_paths)

    def __getitem__(self, index):

        seism_3ch = load_seism_from_npy(self.seism_paths, index)
        geomodel_3ch = load_geomodel_from_npy(self.geomodel_paths, index)

        if self.transform is not None:
            augmented = self.transform(image=geomodel_3ch, seism=seism_3ch)
            geomodel_3ch = augmented["image"]
            seism_3ch = augmented["seism"]

        return seism_3ch, geomodel_3ch
