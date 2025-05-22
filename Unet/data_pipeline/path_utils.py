import os
import re

import numpy as np


def load_model_paths(BASE_PATH):

    model_base_path = os.path.join(BASE_PATH, "configs")
    model_folders = sorted(os.listdir(model_base_path))
    model_nums = [int(folder[7:]) for folder in model_folders]

    # Формируем пути для моделей (rho, vp, vs)
    model_paths = np.array(
        [
            [
                os.path.join(model_base_path, folder, f"{prefix}_{num}.bin")
                for folder, num in zip(model_folders, model_nums)
            ]
            for prefix in ["rho", "vp", "vs"]
        ]
    ).T

    return model_paths


def load_seism_paths(BASE_PATH):

    seism_base_path = os.path.join(BASE_PATH, "seismograms")
    seism_folders = sorted(os.listdir(seism_base_path))
    model_base_path = os.path.join(BASE_PATH, "configs")
    model_folders = sorted(os.listdir(model_base_path))
    model_nums = [int(folder[7:]) for folder in model_folders]

    seism_paths = np.array(
        [
            [
                os.path.join(seism_base_path, folder, f"seismogram_{num}_{i}", "seismogram.txt")
                for folder, num in zip(seism_folders, model_nums)
            ]
            for i in range(3)
        ]
    ).T

    return seism_paths


def load_fault_paths(BASE_PATH):

    model_base_path = os.path.join(BASE_PATH, "configs")
    model_folders = sorted(os.listdir(model_base_path))
    model_nums = [int(folder[7:]) for folder in model_folders]

    fault_paths = np.array(
        [
            os.path.join(model_base_path, folder, f"fault_map_{num}.bin")
            for folder, num in zip(model_folders, model_nums)
        ]
    )

    return fault_paths


def generate_lists_split(base_path, train_ratio=0.85):
    npy_dir = f"{base_path}/npy"
    with open(f"{npy_dir}/file_list.txt") as f:
        lines = f.readlines()

    indices = np.arange(len(lines))
    np.random.shuffle(indices)

    split = int(train_ratio * len(lines))
    train_idx, val_idx = indices[:split], indices[split:]

    train_list = [lines[i].strip() for i in train_idx]
    val_list = [lines[i].strip() for i in val_idx]

    return train_list, val_list
