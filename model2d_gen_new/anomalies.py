import random
from typing import List, Tuple

import numpy as np
from matplotlib.path import Path
from model import GeoModel
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial import ConvexHull


class GeoStructure:
    """
    Базовый абстрактный класс для геологических структур.
    Определяет метод mask_func, который возвращает маску области структуры.
    """

    def mask_func(self, y, x):
        raise NotImplementedError


class GeoStructureGenerator:
    """
    Абстрактный класс генератора одной геологической структуры.
    """

    def generate(self) -> GeoStructure:
        raise NotImplementedError


class GeoStructureListGenerator:
    """
    Абстрактный класс генератора списка геологических структур.
    """

    def generate(self) -> List[GeoStructure]:
        raise NotImplementedError


class AnomalySelector(GeoStructureGenerator):
    """
    Случайным образом выбирает один генератор из списка и возвращает его результат.
    """

    def __init__(self, generators: List[GeoStructureGenerator]):
        self.generators = generators

    def generate(self) -> GeoStructure:
        generator = random.choice(self.generators)
        return generator.generate()


class GeoStructureApplicator:
    """
    Применяет структуру GeoStructure к модели, используя маску и заданное значение.
    """

    def __init__(
        self,
        model: GeoModel,
    ):
        self.model = model
        self.y_mesh, self.x_mesh = np.meshgrid(
            np.arange(model.data.shape[0]), np.arange(model.data.shape[1]), indexing="ij"
        )
        self.y_mesh, self.x_mesh = model.buffered_to_model_coords(self.y_mesh, self.x_mesh)

    def apply_geo_structure(self, anomaly: GeoStructure, model_value):
        mask = anomaly.mask_func(self.y_mesh, self.x_mesh)
        self.model.data[mask] = model_value


# слои как и аномалии наследники GeoStructure из-за схожести в их применении
class LayerAnomaly(GeoStructure):
    """
    Представляет слой, начинающийся с заданной глубины и искривлённый функцией.
    """

    def __init__(self, line_func: "function", start):
        self.line_func = line_func
        self.start = start

    def mask_func(self, y, x):
        return y > self.start + self.line_func(x)


class LayersGenerator(GeoStructureListGenerator):
    """
    Генерирует несколько объектов класса LayerAnomaly.
    """

    def __init__(
        self,
        model: GeoModel,
        line_func: "function",
        num_layers_range: Tuple[int, int],
        layer_thickness_range: Tuple[int, int],
    ):
        self.model = model
        self.num_layers_range = num_layers_range
        self.layer_thickness_range = layer_thickness_range
        self.line_func = line_func

    def generate(self):
        num_layers = np.random.randint(*self.num_layers_range)
        self.model.num_layers = num_layers
        layer_thicknesses = np.random.uniform(
            0.7 * self.model.y_size / num_layers, 1.4 * self.model.y_size / num_layers, num_layers - 1
        )  # -1 потому что первый слой уже есть - это фон
        layer_depths = np.cumsum(layer_thicknesses).astype(int)
        layer_depths[layer_depths >= self.model.y_size] = self.model.y_size - 1  # Ограничение по глубине

        return [LayerAnomaly(self.line_func, start=layer_depths[i]) for i in range(len(layer_thicknesses))]


class DistortedLayerGenerator(GeoStructureGenerator):
    """
    Генератор слоя с искажениями. Основной наклон задаётся через точку старта и угол наклона.
    Искажения добавляются с помощью сплайна.
    """

    def __init__(
        self,
        model: GeoModel,
        line_func: "function",
        distort_range: Tuple[float, float],
        num_points: int,
        y_center_range: Tuple[int, int],
        x_center_range: Tuple[int, int],
        angle_rad_range: Tuple[float, float],  # в числах пи
    ):
        self.model = model
        self.line_func = line_func
        self.distort_range = distort_range
        self.num_points = num_points
        self.y_center_range = y_center_range
        self.x_center_range = x_center_range
        self.angle_rad_range = angle_rad_range

    def generate(self):
        angle_rad = np.random.uniform(*self.angle_rad_range) * np.random.choice([-1, 1])
        y_center = np.random.randint(*self.y_center_range)
        x_center = np.random.randint(*self.x_center_range)

        slope = np.tan(angle_rad)

        x_points = np.linspace(0, self.model.x_size, num=self.num_points)
        y_points = np.random.uniform(*self.distort_range, size=self.num_points)
        spline = CubicSpline(x_points, y_points)

        def layer_func(x):
            base = y_center + slope * (x - x_center) + self.line_func(x)
            return base + spline(x)

        return LayerAnomaly(layer_func, start=0)


class EllipseAnomaly(GeoStructure):
    def __init__(self, cx, cy, rx, ry):
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry

    def mask_func(self, y, x):
        return ((x - self.cx) / self.rx) ** 2 + ((y - self.cy) / self.ry) ** 2 <= 1


class EllipseGenerator(GeoStructureGenerator):
    def __init__(
        self,
        cy_range: Tuple[int, int],
        cx_range: Tuple[int, int],
        ry_range: Tuple[int, int],
        rx_range: Tuple[int, int],
    ):
        self.cy_range = cy_range
        self.cx_range = cx_range
        self.ry_range = ry_range
        self.rx_range = rx_range

    def generate(self):
        cy = np.random.randint(*self.cy_range)
        cx = np.random.randint(*self.cx_range)
        ry = np.random.randint(*self.ry_range)
        rx = np.random.randint(*self.rx_range)
        return EllipseAnomaly(cx, cy, rx, ry)


class SplineShapeAnomaly(GeoStructure):
    """
    Замкнутая область произвольной формы, построенная по сплайну по выпуклой оболочке точек.
    """

    def __init__(self, x_points: np.ndarray, y_points: np.ndarray):
        assert len(x_points) == len(y_points) >= 3, "Нужно минимум 3 точки для замкнутой формы"

        self.x_points = np.append(x_points, x_points[0])
        self.y_points = np.append(y_points, y_points[0])

        t = np.linspace(0, 1, len(self.x_points))
        self.x_spline = CubicSpline(t, self.x_points, bc_type="not-a-knot")
        self.y_spline = CubicSpline(t, self.y_points, bc_type="not-a-knot")

        t_dense = np.linspace(0, 1, 1000)
        x_dense = self.x_spline(t_dense)
        y_dense = self.y_spline(t_dense)
        self.path = Path(np.column_stack([x_dense, y_dense]))

    def mask_func(self, y, x):
        coords = np.vstack([x.ravel(), y.ravel()]).T
        mask = self.path.contains_points(coords).reshape(x.shape)
        return mask


class SplineGenerator(GeoStructureGenerator):
    """
    Генерирует произвольную выпуклую область в виде SplineShapeAnomaly.
    """

    def __init__(
        self,
        num_points: int,
        cy_range: Tuple[int, int],
        cx_range: Tuple[int, int],
        ry_range: Tuple[int, int],
        rx_range: Tuple[int, int],
    ):
        self.num_points = num_points
        self.cy_range = cy_range
        self.cx_range = cx_range
        self.ry_range = ry_range
        self.rx_range = rx_range

    def generate(self):
        center_y = np.random.randint(*self.cy_range)
        center_x = np.random.randint(*self.cx_range)
        ry = np.random.randint(*self.ry_range)
        rx = np.random.randint(*self.rx_range)

        # Генерируем случайные точки внутри прямоугольника
        x_offsets = np.random.uniform(-rx, rx, size=self.num_points * 3)
        y_offsets = np.random.uniform(-ry, ry, size=self.num_points * 3)
        points = np.stack([x_offsets, y_offsets], axis=1)

        # Строим выпуклую оболочку
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Центрируем вокруг центра
        x_points = hull_points[:, 0] + center_x
        y_points = hull_points[:, 1] + center_y

        return SplineShapeAnomaly(x_points, y_points)


class MountainGenerator(GeoStructureGenerator):
    """
    Генерирует произвольную MountainAnomaly.
    """

    def __init__(
        self,
        num_points: int,
        y_start: int,
        cx_range: Tuple[int, int],
        ry_range: Tuple[int, int],
        rx_range: Tuple[int, int],
    ):
        self.num_points = num_points
        self.y_start = y_start
        self.cx_range = cx_range
        self.ry_range = ry_range
        self.rx_range = rx_range

    def generate(self):
        center_x = np.random.randint(*self.cx_range)
        ry = np.random.randint(*self.ry_range)
        rx = np.random.randint(*self.rx_range)

        x_points = np.linspace(center_x - rx, center_x + rx, num=self.num_points)
        y_points = np.random.uniform(0, ry, size=self.num_points)
        y_points[0], y_points[-1] = 0, 0

        # Строим "замкнутую гору": верхняя часть по сплайну, нижняя — горизонтальная линия
        x_full = np.append(x_points, x_points[::-1])
        y_full = np.append(self.y_start - y_points, np.full_like(y_points, self.y_start))

        return SplineShapeAnomaly(x_full, y_full)


class ImageGenerator(GeoStructureGenerator):
    """
    Аномалия, использующая в качестве контура spline

    xy: двумерный массив, заранее подготовленный контур изображения
    """

    def __init__(
        self,
        xy: np.ndarray,
        cy_range: Tuple[int, int],
        cx_range: Tuple[int, int],
        ry_range: Tuple[int, int],
        rx_range: Tuple[int, int],
    ):
        self.xy = xy
        self.cy_range = cy_range
        self.cx_range = cx_range
        self.ry_range = ry_range
        self.rx_range = rx_range

    def generate(self):
        center_y = np.random.randint(*self.cy_range)
        center_x = np.random.randint(*self.cx_range)
        ry = np.random.randint(*self.ry_range)
        rx = np.random.randint(*self.rx_range)

        xy_scaled = np.copy(self.xy)
        xy_scaled[:, 0] *= rx
        xy_scaled[:, 1] *= ry
        xy_scaled[:, 0] += center_x - rx / 2
        xy_scaled[:, 1] += center_y - ry / 2

        x_points, y_points = xy_scaled[:, 0], xy_scaled[:, 1]

        return SplineShapeAnomaly(x_points, y_points)
