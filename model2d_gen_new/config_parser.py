from typing import List, Union

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
from layer_functions import PerlinLine, ragged_line, zero_line
from model import GeoModel
from physical import PhysicalModelBuilder


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

        line_func_name = layers_cfg["line_func"]
        if line_func_name == "ragged":
            line_func = ragged_line
        elif line_func_name == "perlin":
            line_func = PerlinLine()
        elif line_func_name == "zero":
            line_func = zero_line
        else:
            raise ValueError(f"Unknown line_func: {line_func_name}")

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

        return FaultsGenerator(
            num_faults_range=tuple(faults_cfg["num_faults_range"]),
            displacement_range=tuple(faults_cfg["displacement_range"]),
            y_center_range=tuple(faults_cfg["y_center_range"]),
            x_center_range=tuple(faults_cfg["x_center_range"]),
            angle_rad_range=tuple(np.pi * np.array(faults_cfg["angle_rad_range"])),
        )

    def create_anomaly_selector(self, model: GeoModel) -> Union[AnomalySelector, None]:
        if not self.config.get("anomalies"):
            return None

        generators: List[GeoStructureGenerator] = []

        for anomaly_cfg in self.config["anomalies"]:
            type_ = anomaly_cfg["type"]

            if type_ == "ellipse":
                generator = EllipseGenerator(
                    cy_range=tuple(anomaly_cfg["cy_range"]),
                    cx_range=tuple(anomaly_cfg["cx_range"]),
                    ry_range=tuple(anomaly_cfg["ry_range"]),
                    rx_range=tuple(anomaly_cfg["rx_range"]),
                )
            elif type_ == "spline":
                generator = SplineGenerator(
                    num_points=anomaly_cfg["num_points"],
                    cy_range=tuple(anomaly_cfg["cy_range"]),
                    cx_range=tuple(anomaly_cfg["cx_range"]),
                    ry_range=tuple(anomaly_cfg["ry_range"]),
                    rx_range=tuple(anomaly_cfg["rx_range"]),
                )
            elif type_ == "image":
                image_path = anomaly_cfg["file"]
                xy = np.load(image_path)
                generator = ImageGenerator(
                    xy=xy,
                    cy_range=tuple(anomaly_cfg["cy_range"]),
                    cx_range=tuple(anomaly_cfg["cx_range"]),
                    ry_range=tuple(anomaly_cfg["ry_range"]),
                    rx_range=tuple(anomaly_cfg["rx_range"]),
                )
            elif type_ == "mountain":
                generator = MountainGenerator(
                    num_points=anomaly_cfg["num_points"],
                    y_start=anomaly_cfg["y_start"],
                    cx_range=tuple(anomaly_cfg["cx_range"]),
                    ry_range=tuple(anomaly_cfg["ry_range"]),
                    rx_range=tuple(anomaly_cfg["rx_range"]),
                )
            elif type_ == "distorted_layer":
                line_func_name = anomaly_cfg["line_func"]
                if line_func_name == "ragged":
                    line_func = ragged_line
                elif line_func_name == "perlin":
                    line_func = PerlinLine()
                elif line_func_name == "zero":
                    line_func = zero_line
                else:
                    raise ValueError(f"Unknown line_func: {line_func_name}")
                generator = DistortedLayerGenerator(
                    model=model,
                    line_func=line_func,
                    num_points=anomaly_cfg["num_points"],
                    angle_rad_range=tuple(anomaly_cfg["angle_rad_range"]),
                    y_center_range=tuple(anomaly_cfg["y_center_range"]),
                    x_center_range=tuple(anomaly_cfg["x_center_range"]),
                )
            else:
                raise ValueError(f"Unknown anomaly type: {type_}")

            generators.append(generator)

        return AnomalySelector(generators)

    def create_physical_model_builder(self) -> PhysicalModelBuilder:
        physics_cfg = self.config.get("physics")
        if physics_cfg is None:
            raise ValueError("Missing physics configuration")

        return PhysicalModelBuilder(
            rho_range=tuple(physics_cfg["rho_range"]),
            vp_range=tuple(physics_cfg["vp_range"]),
            delimiter_range=tuple(physics_cfg["delimiter_range"]),
            multiplicator_range=tuple(physics_cfg["multiplicator_range"]),
        )
