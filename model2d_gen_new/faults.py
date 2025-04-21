import numpy as np
from typing import Tuple

from model import GeoModel


class Fault:
    def __init__(self, displacement_module, y_center, x_center, angle_rad):
        """
        Представляет геологический разлом как наклонную линию со смещением.

        Атрибуты:
            displacement_module (int): Модуль вектора смещения вдоль линии разлома.
            y_center (int), x_center (int): Центр разлома
            angle_rad (float): Угол наклона разлома в радианах, против часовой от оси X.
            (не стоит забывать, что в наших координатах y это глубина)
        """
        self.displacement_module = displacement_module
        self.y_center = y_center
        self.x_center = x_center
        self.angle_rad = angle_rad

    def mask_func(self, y, x):
        dx = x - self.x_center
        dy = y - self.y_center
        vx = np.cos(self.angle_rad)
        vy = np.sin(self.angle_rad)
        cross = vx * dy - vy * dx
        return cross < 0  # True, если справа

    def distance_to_line(self, y, x):
        dx = x - self.x_center
        dy = y - self.y_center
        return np.abs(-np.sin(self.angle_rad) * dx + np.cos(self.angle_rad) * dy)


class FaultsGenerator:
    def __init__(
        self,
        num_faults_range: Tuple[int, int],
        displacement_range: Tuple[int, int],
        y_center_range: Tuple[int, int],
        x_center_range: Tuple[int, int],
        angle_rad_range: Tuple[float, float],  # в числах пи
    ):
        self.num_faults_range = num_faults_range
        self.displacement_range = displacement_range
        self.y_center_range = y_center_range
        self.x_center_range = x_center_range
        self.angle_rad_range = angle_rad_range

    def generate(self):
        self.num_faults = np.random.randint(*self.num_faults_range)

        self.faults = []

        for i in range(self.num_faults):
            displacement_module = np.random.randint(*self.displacement_range) * np.random.choice([-1, 1])
            y_center = np.random.randint(*self.y_center_range)
            x_center = np.random.randint(*self.x_center_range)
            angle_rad = np.random.uniform(*self.angle_rad_range) * np.random.choice([-1, 1])
            fault = Fault(displacement_module, y_center, x_center, angle_rad)
            self.faults.append(fault)

        return self.faults


class FaultApplicator:
    def __init__(self, model: GeoModel):
        self.model = model
        self.y_mesh, self.x_mesh = np.meshgrid(
            np.arange(model.data.shape[0]), np.arange(model.data.shape[1]), indexing="ij"
        )
        self.y_mesh, self.x_mesh = model.buffered_to_model_coords(self.y_mesh, self.x_mesh)
    
    def draw_fault(self, fault: Fault):
        """
        рисует контур разлома на model.fault_data
        """
        mask = fault.distance_to_line(self.y_mesh, self.x_mesh) < 2.0
        self.model.faults_data[mask] = 1.0
        #self.model.data[mask] = 3.0

    def apply_fault(self, fault: Fault):
        displacement_x = (np.cos(fault.angle_rad) * fault.displacement_module).astype(int)
        displacement_y = (np.sin(fault.angle_rad) * fault.displacement_module).astype(int)
        mask = fault.mask_func(self.y_mesh, self.x_mesh)
        self.y_mesh, self.x_mesh = self.model.model_to_buffered_coords(self.y_mesh, self.x_mesh)
        src_y = np.clip(self.y_mesh[mask] + displacement_y, 0, self.model.data.shape[0] - 1).astype(int)
        src_x = np.clip(self.x_mesh[mask] + displacement_x, 0, self.model.data.shape[1] - 1).astype(int)
        self.model.faults_data[mask] = self.model.faults_data[src_y, src_x]
        self.model.data[mask] = self.model.data[src_y, src_x]
        self.y_mesh, self.x_mesh = self.model.buffered_to_model_coords(self.y_mesh, self.x_mesh)

        self.draw_fault(fault)
