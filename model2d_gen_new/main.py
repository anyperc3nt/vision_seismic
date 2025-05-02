import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from anomalies import GeoStructureApplicator
from config_parser import ConfigParser
from faults import FaultApplicator
from rect_config_generator import rect_config_generator
from tqdm import tqdm


# Визуализация распределений rho, vp и vs
def save_plot(data, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data,
        cmap="viridis",
    )  # norm="log")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# Запись распределений rho, vp и vs в бинарники
def save_to_bin(data, filename):
    data_s = np.flip(data, axis=0)
    data_s.astype("f").tofile(filename)


def model_generator(config_path, dataset_path):
    parser = ConfigParser(config_path)

    n_samples = parser.get_num_samples()

    model = parser.create_model()
    applicator = GeoStructureApplicator(model)
    layers_gen = parser.create_layers_generator(model)
    faults_gen = parser.create_faults_generator()
    anomaly_selector = parser.create_anomaly_selector(model)
    physical_model_builder = parser.create_physical_model_builder()

    for num in tqdm(
        range(n_samples),
        desc=f"{config_path:<40}",
    ):
        model.clear()

        if not os.path.exists(f"{dataset_path}/models/model_{num}"):
            os.makedirs(f"{dataset_path}/models/model_{num}", exist_ok=True)

        if not os.path.exists(f"{dataset_path}/configs/config_{num}"):
            os.makedirs(f"{dataset_path}/configs/config_{num}", exist_ok=True)

        # Генерация и применение слоёв
        if layers_gen is not None:
            layers = layers_gen.generate()
            for i, layer in enumerate(layers):
                applicator.apply_geo_structure(layer, model_value=i + 2)

        # Генерация и применение разломов
        if faults_gen is not None:
            faults = faults_gen.generate()
            fault_applicator = FaultApplicator(model)
            for fault in faults:
                fault_applicator.apply_fault(fault)

        # Генерация и применение аномалии
        if anomaly_selector is not None:
            anomaly = anomaly_selector.generate()
            model_value = model.num_layers + 1
            applicator.apply_geo_structure(anomaly, model_value)

        # Генерация физических параметров модели (rho, vp, vs)
        rho_model, vp_model, vs_model = physical_model_builder.build_maps(model.workview, model.num_layers)

        if args.save_plots:
            save_plot(rho_model, f"{dataset_path}/models/model_{num}/rho.png")
            save_plot(vp_model, f"{dataset_path}/models/model_{num}/vp.png")
            save_plot(vs_model, f"{dataset_path}/models/model_{num}/vs.png")
            if faults_gen is not None:
                save_plot(model.faults_workview[...], f"{dataset_path}/models/model_{num}/fault_map.png")

        save_to_bin(rho_model, f"{dataset_path}/configs/config_{num}/rho_{num}.bin")
        save_to_bin(vp_model, f"{dataset_path}/configs/config_{num}/vp_{num}.bin")
        save_to_bin(vs_model, f"{dataset_path}/configs/config_{num}/vs_{num}.bin")
        if faults_gen is not None:
            save_to_bin(model.faults_workview[...], f"{dataset_path}/configs/config_{num}/fault_map_{num}.bin")

        for j in range(3):
            rect_config_generator(dataset_path, num, j)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Для датасета нам не нужно сохранять png картинки, + это в разы замедляет выполнение
    parser.add_argument("--save_plots", action="store_true")
    args = parser.parse_args()

    if args.save_plots:
        print("Сохраняем картинки распределений")

    for filename in os.listdir("model_configs"):
        if filename.endswith(".yaml"):
            name = os.path.splitext(filename)[0]
            dataset_path = os.path.join("dataset", name)
            config_path = os.path.join("model_configs", filename)
            # os.makedirs(dataset_path, exist_ok=True)
            model_generator(config_path, dataset_path)
