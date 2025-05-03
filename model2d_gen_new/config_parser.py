import inspect
from typing import List, Tuple, Union

import numpy as np
import yaml
from anomalies import (
    AnomalySelector,
    DistortedLayerGenerator,
    EllipseGenerator,
    GeoStructure,
    GeoStructureGenerator,
    GeoStructureListGenerator,
    ImageGenerator,
    LayersGenerator,
    MountainGenerator,
    SplineGenerator,
)
from faults import FaultsGenerator
from layer_functions import PerlinLine, RaggedLine, ZeroLine
from model import GeoModel
from physical import PhysicalModelBuilder


def create_instance(cls, config_dict):
    """
    Для умного чтения конфигов без рутинного прописывания параметров.

    Есть проверка на параметры, такие как 'angle_rad_range',
    которые нужно умножить на np.pi.
    """
    sig = inspect.signature(cls)
    kwargs = {}

    for name, param in sig.parameters.items():
        if name in config_dict:
            value = config_dict[name]
            if "angle_rad" in name and isinstance(value, list):
                value = np.array(value) * np.pi  # Если это angle_rad_range, умножаем на np.pi

            # Если значение списка, конвертируем в tuple
            if isinstance(value, list) and param.annotation in (tuple, Tuple):
                value = tuple(value)

            kwargs[name] = value

    return cls(**kwargs)


class ConfigParser:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_num_samples(self) -> int:
        num_samples = self.config.get("num_samples")
        if num_samples is None:
            raise ValueError("Missing num_samples")
        return num_samples

    def create_model(self) -> GeoModel:
        x_size = self.config["model"]["x_size"]
        y_size = self.config["model"]["y_size"]
        # вычисляем buffer по числу разломов и смещению
        faults_cfg = self.config.get("faults")
        if faults_cfg:
            max_faults = faults_cfg["num_faults_range"][1]
            max_disp = faults_cfg["displacement_range"][1]
            buffer = int(max_faults * max_disp)
        else:
            buffer = 0
        return GeoModel(y_size=y_size, x_size=x_size, buffer=buffer)

    def create_layers_generator(self, model: GeoModel) -> Union[LayersGenerator, None]:
        layers_cfg = self.config.get("layers")
        if layers_cfg is None:
            return None

        line_func = globals()[layers_cfg["line_func"]]()

        return LayersGenerator(
            model=model,
            line_func=line_func,
            num_layers_range=tuple(layers_cfg["num_layers_range"]),
            layer_thickness_range=tuple(layers_cfg["layer_thickness_range"]),
        )

    def create_faults_generator(self) -> Union[FaultsGenerator, None]:
        faults_cfg = self.config.get("faults")
        if faults_cfg is None:
            return None

        return create_instance(FaultsGenerator, faults_cfg)

    def create_anomaly_selector(self, model: GeoModel) -> Union[AnomalySelector, None]:
        if not self.config.get("anomalies"):
            return None

        generators = []

        for cfg in self.config["anomalies"]:
            type_ = cfg["type"]

            if type_ == "distorted_layer":
                line_func = globals()[cfg["line_func"]]()
                cfg = {**cfg, "line_func": line_func, "model": model}
                generator = create_instance(DistortedLayerGenerator, cfg)

            elif type_ == "image":
                xy = np.load(cfg["file"])
                cfg = {**cfg, "xy": xy}
                generator = create_instance(ImageGenerator, cfg)

            elif type_ == "ellipse":
                generator = create_instance(EllipseGenerator, cfg)

            elif type_ == "spline":
                generator = create_instance(SplineGenerator, cfg)

            elif type_ == "mountain":
                generator = create_instance(MountainGenerator, cfg)

            else:
                raise ValueError(f"Unknown anomaly type: {type_}")

            generators.append(generator)

        return AnomalySelector(generators)

    def create_physical_model_builder(self) -> PhysicalModelBuilder:
        physics_cfg = self.config.get("physics")
        if physics_cfg is None:
            raise ValueError("Missing physics configuration")

        return create_instance(PhysicalModelBuilder, physics_cfg)
