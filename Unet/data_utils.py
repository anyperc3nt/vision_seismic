import os

import numpy as np


def try_load(path):
    try:
        data = np.loadtxt(path, dtype="f")
        return True
    except (OSError, ValueError):
        return False


def correct_(paths):
    """F
    проверка датасета на папки, где нехватает данных (например, недосчитаны сейсмограммы)
    """
    # эта операция достаточно долгая, по-этому если она нужна я проделываю ее 1 раз и сохраняю результат в csv
    if try_load("correct_data.csv"):
        correct_data = np.loadtxt("correct_data.csv", dtype=bool)
    else:
        correct_data = np.array([try_load(path[2]) for path in paths], dtype=bool)
        np.savetxt("correct_data.csv", correct_data, delimiter=",", fmt="%d")
    return correct_data


def load_model_paths(BASE_PATH="../../model_2d_faults_03_2025/dataset"):

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

    print(f"loaded {model_paths.shape} paths")
    print(f"example:\n{model_paths[0]}")

    return model_paths


def load_seism_paths(BASE_PATH="../../model_2d_faults_03_2025/dataset"):

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

    print(f"loaded {seism_paths.shape} paths")
    print(f"example:\n{seism_paths[0]}")

    return seism_paths


def load_fault_paths(BASE_PATH="../../model_2d_faults_03_2025/dataset"):

    model_base_path = os.path.join(BASE_PATH, "configs")
    model_folders = sorted(os.listdir(model_base_path))
    model_nums = [int(folder[7:]) for folder in model_folders]

    # Формируем пути для моделей (rho, vp, vs)
    fault_paths = np.array(
        [
            os.path.join(model_base_path, folder, f"fault_map_{num}.bin")
            for folder, num in zip(model_folders, model_nums)
        ]
    )

    print(f"loaded {fault_paths.shape} paths")
    print(f"example:\n{fault_paths[0]}")

    return fault_paths
