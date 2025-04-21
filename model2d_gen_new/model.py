import numpy as np
from typing import Tuple


"""
чтобы избежать артефактов, связанных с тем, что при применении сдвига негде брать
информацию о значениях модели за краями массива, модель генерируется с "запасом",
равным buffer = max_num_faults * max_displacement с каждого края.
"""


class GeoModel:
    """
    Геологическая модель с буфером по краям, хранящая значения индексов.

    Аргументы конструктора:
        y_size (int), x_size (int): Высота, ширина рабочей области модели (без учёта буфера).
        buffer (int): Размер буфера по краям модели.

    Атрибуты:
        data (np.ndarray): Полная сетка модели.
        workview (np.ndarray): Рабочая часть модели без буферов.
        buffered (np.ndarray): То же, что и data.
        num_layers (int): Количество слоёв (по умолчанию 1).

    Методы:
        buffered_to_model_coords(y, x) -> (y, x): Преобразует координаты из буферной в модельную систему.
        model_to_buffered_coords(y, x) -> (y, x): Преобразует координаты из модельной в буферную систему.
    """

    def __init__(self, y_size, x_size, buffer):
        self.buffer = buffer
        self.y_size = y_size
        self.x_size = x_size

        self.data = np.zeros((y_size + 2 * buffer, x_size + 2 * buffer), dtype=np.float32) + 1.0
        self.workview = self.data[buffer : -buffer or None, buffer : -buffer or None]
        self.buffered = self.data

        self.faults_data = np.zeros((y_size + 2 * buffer, x_size + 2 * buffer), dtype=np.float32)
        self.faults_workview = self.faults_data[buffer : -buffer or None, buffer : -buffer or None]

        self.num_layers = 1  # нужно, тк кк слоев всегда минимум 1
    
    def clear(self):
        self.data.fill(1.0)
        self.num_layers = 1
        self.faults_data.fill(0.0)

    def buffered_to_model_coords(self, y, x):
        return y - self.buffer, x - self.buffer

    def model_to_buffered_coords(self, y, x):
        return y + self.buffer, x + self.buffer
