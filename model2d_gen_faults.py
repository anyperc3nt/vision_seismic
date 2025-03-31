import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
import math
import os
from perlin_noise import PerlinNoise
import random

# seed
seed = int(np.random.SeedSequence().generate_state(1)[0])
#seed = 676404499

print(f"seed = {seed}")
np.random.seed(seed)
random.seed(seed)

class Fault:
    """
    следующие параметры задают разлом::
    x1, x2, y1, y2 : координаты начала и конца разлома
    shift_distance : Величина сдвига разлома.
    apply_above : {1, -1} - определяет, сдвигаться будет часть модели выше или ниже разлома.

    вспомогательные параметры, вычисляются из заданных
    dx, dy Горизонтальная и вертикальная составляющая сдвига.
    """
    x1: int
    x2: int
    y1: int
    y2: int
    dx: int
    dy : int
    apply_above: int
    shift_distance: float

    def __init__(self, shift_min: float, shift_max: float, padding):
        """
        shift_min, shift_max : минимум и максимум shift_distance.
        padding : отступ для координат, связанный с тем, что модель размера x_size + 2*padding, y_size + 2*padding.
        """
        self.x1 = random.randint(0, 500) + padding
        self.x2 = random.randint(0, 500) + padding
        self.x2 += 1 * (self.x1==self.x2) #отбрасываем вертикальные прямые
        self.y1 = 0 + padding
        self.y2 = 200 + padding
        self.apply_above = random.choice([1,-1])
        self.shift_distance = (random.random() * 2 - 1) * ( (shift_max - shift_min) + shift_min )
        self.dx = int(
            self.shift_distance
            * (self.x2 - self.x1)
            / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5
        )
        self.dy = int(
            self.shift_distance
            * (self.y2 - self.y1)
            / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5
        )
    
    def dot_in_area(self, x, y):
        """
        Проверяет, находится ли точка (x, y) в области действия разлома
        """
        return (y - (x - self.x1) * (self.y2 - self.y1) / (self.x2 - self.x1) - self.y1) * self.apply_above > 0

    def distance(self, x, y):
        """
        Вычисляет расстояние от точки (x, y) до линии разлома.       
        """
        return 1.0* np.abs((self.y2 - self.y1) * x - (self.x2 - self.x1) * y + self.x2 * self.y1 - self.y2 * self.x1) / ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5


def model_generator(num):
    noise1 = PerlinNoise(octaves=3)
    noise2 = PerlinNoise(octaves=6)
    noise3 = PerlinNoise(octaves=12)

    def ragged_line(x):
        return (
            noise1(x / 500) * 40
            + noise2(x / 500) * 20
            + noise3(x / 500) * 10
            + np.random.uniform(-1.5, 1.5)
        )

    if not os.path.exists(f"./dataset/models/model_{num}"):
        os.makedirs(f"./dataset/models/model_{num}", exist_ok=True)

    if not os.path.exists(f"./dataset/configs/config_{num}"):
        os.makedirs(f"./dataset/configs/config_{num}", exist_ok=True)

    # Параметры модели
    x_size, y_size = 501, 201  # Размеры модели (длина и глубина)
    H = 200  # Глубина в метрах
    num_layers = np.random.randint(3, 11)  # Случайное количество слоев
    num_layers = 11

    # Параметры фолтов
    shift_min = 7
    shift_max = 30
    num_faults = 12#np.random.randint(1, 5) #Случайное количество фолтов

    # Параметры отступа, связанные с генерацией фолтов
    padding = num_faults * shift_max
    x_size += padding * 2
    y_size += padding * 2
    """
    чтобы избежать артефактов, связанных с тем, что при применении сдвига негде брать
    информацию о значениях модели за краями массива, модель генерируется с "запасом",
    равным padding с каждого края. затем, после применения всех разломов, модель
    обрезается до необходимого размера
    """

    # Диапазоны параметров слоев
    rho_min, rho_max = 2000.0, 5000.0
    vp_min, vp_max = 2200.0, 5000.0  # км/с
    delimiter = np.random.uniform(2.0, 3.0)
    multiplicator = np.random.uniform(1.1, 1.15)

    # Генерация параметров слоев
    layer_thicknesses = np.random.uniform(
        0.7 * H / num_layers, 1.4 * H / num_layers, num_layers
    )
    layer_depths = np.cumsum(layer_thicknesses).astype(int) + padding
    layer_depths[layer_depths >= y_size] = y_size - 1  # Ограничение по глубине

    rho = [np.random.uniform(rho_min, rho_min + 1.5)]
    vp = [rho[0] * multiplicator]
    vs = [vp[0] / delimiter]

    for i in range(1, num_layers):
        temp_rho = rho[i - 1] * np.random.uniform(1, 2.4 ** (1 / num_layers))
        rho.append(temp_rho if temp_rho < rho_max else rho_max)
        temp_vp = rho[i] * multiplicator
        vp.append(temp_vp if temp_vp < vp_max else vp_max)
        vs.append(vp[i] / delimiter)

    # Создание 2D модели
    model = np.zeros((x_size, y_size))
    print(f"num_faults: {num_faults}")

    for i in range(num_layers):
        start = layer_depths[i - 1] if i > 0 else 0
        # end = layer_depths[i]
        for x in range(x_size):
            if start == 0:
                model[x, :] = i + 1
            # elif num_layers - 1 == i:
            #     model[x, int(start + ragged_line(x-padding)):] = i + 1
            else:
                model[x, int(start + ragged_line(x-padding)) :] = i + 1
    
    # Далее часть, связанная с разломами
    # Меш для вычисления булевых масок
    x_mesh, y_mesh = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing='ij')
    # Массив для хранения булевых масок
    mask = np.zeros_like(model)
    # Модель на которой отображаются только разломы
    model_faults = np.zeros((x_size, y_size))

    def apply_fault(f: Fault, model, mask):
        """
        применяет действие разлома на массив model
        mask - массив для хранения булевых масок, передается чтобы не тратить время на создание каждый раз
        """
        pad = max(np.abs(f.dx), np.abs(f.dy))
        model_padded = np.pad(model, pad_width=pad, mode="edge")
        # Вычисляем булеву маску для всей области
        mask = f.dot_in_area(x_mesh,y_mesh)
        model[mask] = model_padded[x_mesh[mask] + pad - f.dx, y_mesh[mask] + pad - f.dy]
        return model

    def draw_fault(f: Fault, model, mask):
        """
        рисует контур разлома на model
        mask - массив для хранения булевых масок, передается чтобы не тратить время на создание каждый раз
        """
        mask = f.distance(x_mesh,y_mesh) < 1.5
        model[mask] = num_layers + 5
        return model
    
    # Создание и применение разломов
    if True:
        plt.figure(figsize=(30, 15))
        faults = [Fault(shift_min, shift_max, padding) for i in range(num_faults)]
        
        for i, fault in enumerate(faults):
            print(f"вверху/внизу: {fault.apply_above}, модуль: {round(fault.shift_distance,3)}, dx: {fault.dx}, dy: {fault.dy}")
            model = apply_fault(fault, model, mask)
            model = draw_fault(fault, model, mask)
            model_faults = apply_fault(fault, model_faults, mask)
            model_faults = draw_fault(fault, model_faults, mask)
            plt.subplot(int(np.ceil(num_faults**0.5)), int(np.ceil(num_faults**0.5)), i + 1)
            plt.title(f"fault{i+1}")
            plt.imshow(model[padding : model.shape[0] - padding, padding : model.shape[1] - padding].T)
        plt.axis("off")
        plt.savefig(f"./dataset/models/model_{num}/fault_log.png",bbox_inches="tight",pad_inches=0)
    
    print(f"средний сдвиг по y: {round(np.mean([faults[i].dy for i in range(num_faults)]),3)}")
    print(f"средний сдвиг по x: {round(np.mean([faults[i].dx for i in range(num_faults)]),3)}")
    print(f"средний модуль сдвига: {round(np.mean([faults[i].shift_distance for i in range(num_faults)]),3)}")

    #Обрезка модели
    model = np.copy(model[padding : model.shape[0] - padding, padding : model.shape[1] - padding])
    model_faults = np.copy(model_faults[padding : model_faults.shape[0] - padding, padding : model_faults.shape[1] - padding])

    plt.figure(figsize=(8, 6))
    plt.imshow(model_faults.T, cmap="Greys", interpolation="nearest")
    plt.axis("off")
    plt.savefig(f"./dataset/models/model_{num}/fault_mask.png",bbox_inches="tight",pad_inches=0)
    plt.close()

    # Визуализация распределений rho, vp и vs
    def save_distribution(data, title, filename):
        plt.figure(figsize=(8, 6))
        plt.imshow(data.T, cmap="viridis", norm="log")
        plt.axis("off")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()

    # Распределение rho
    rho_model = np.zeros_like(model, dtype=float)
    for i in range(num_layers + 1):
        if i < len(rho):
            rho_model[model == i + 1] = rho[i]
        else:
            rho_model[model == i + 1] = 4800.0
    rho_model_s = np.flip(rho_model.T, axis=0)
    rho_model_s.astype("f").tofile(f"./dataset/configs/config_{num}/rho_{num}.bin")
    save_distribution(rho_model, "rho", f"./dataset/models/model_{num}/rho.png")

    # Распределение vp
    vp_model = np.zeros_like(model, dtype=float)
    for i in range(num_layers + 1):
        if i < len(vp):
            vp_model[model == i + 1] = vp[i]
        else:
            vp_model[model == i + 1] = 4600.0
    vp_model_s = np.flip(vp_model.T, axis=0)
    vp_model_s.astype("f").tofile(f"./dataset/configs/config_{num}/vp_{num}.bin")
    save_distribution(vp_model, "Vp", f"./dataset/models/model_{num}/vp.png")

    # Распределение vs
    vs_model = np.zeros_like(model, dtype=float)
    for i in range(num_layers + 1):
        if i < len(vs):
            vs_model[model == i + 1] = vs[i]
        else:
            vs_model[model == i + 1] = 2000.0 + (
                (vs[-1] - np.average(vs)) / (vp_max / delimiter)
            )
    vs_model_s = np.flip(vs_model.T, axis=0)
    vs_model_s.astype("f").tofile(f"./dataset/configs/config_{num}/vs_{num}.bin")
    save_distribution(vs_model, "Vs", f"./dataset/models/model_{num}/vs.png")


def config_generator(num1, num2):
    if not os.path.exists(f"./dataset/seismograms/seismogram_{num1}"):
        os.makedirs(f"./dataset/seismograms/seismogram_{num1}", exist_ok=True)

    if not os.path.exists(
        f"./dataset/seismograms/seismogram_{num1}/seismogram_{num1}_{num2}"
    ):
        os.makedirs(
            f"./dataset/seismograms/seismogram_{num1}/seismogram_{num1}_{num2}",
            exist_ok=True,
        )

    x_coord = 100 + 2400 * num2
    config = f"""

        verbose = true

        dt = 0.0015

        steps = 2000


        [grids]
            [grid]
                id = ore_body
                [node]
                    name = ElasticMetaNode2D
                [/node]
                [material_node]
                    name = ElasticMaterialMetaNode
                [/material_node]
                [material]
                    c1 = 1
                    c2 = 1
                    rho = 1
                [/material]
                [factory]
                    name = RectGridFactory
                    size = 501, 201
                    origin = 0, -2000
                    spacing = 10, 10
                [/factory]
                [schema]
                    name = ElasticMatRectSchema2DRusanov3
                [/schema]
                [fillers]
                    [filler]
                        name = RectNoReflectFiller
                        axis = 0
                        side = 0
                    [/filler]
                    [filler]
                        name = RectNoReflectFiller
                        axis = 0
                        side = 1
                    [/filler]
                    [filler]
                        name = RectNoReflectFiller
                        axis = 1
                        side = 0
                    [/filler]
                    [filler]
                        name = RectNoReflectFiller
                        axis = 1
                        side = 1
                    [/filler]
                [/fillers]
                [correctors]
                            [corrector]
                        name = ForceRectElasticBoundary2D
                                axis = 1
                                side = 1
                    [/corrector]

                    [corrector]
                        name = PointSourceCorrector2D
                        coords = {x_coord}, -100, 0.0
                        compression = 1.0
                        axis = 1
                        eps = 2
                        save = ../../seismograms/seismogram_{num1}/source_{num1}_{num2}.vtk
                        gauss_w = 5
                        [impulse]
                            name = FileInterpolationImpulse
                            [interpolator]
                                name = PiceWiceInterpolator1D
                                file = ./dataset/ricker_30.txt
                            [/interpolator]
                        [/impulse]
                    [/corrector]
                [/correctors]

            [/grid]
        [/grids]

        [contacts]
        [/contacts]

        [initials]
            [initial]
                name = StructuredFileLoader
                path = vp_{num1}.bin
                value = c1
                binary = true
                order = 1
            [/initial]
            [initial]
                name = StructuredFileLoader
                path = vs_{num1}.bin
                value = c2
                binary = true
                order = 2
            [/initial]
            [initial]
                name = StructuredFileLoader
                path = rho_{num1}.bin
                value = rho
                binary = true
                order = 3
            [/initial]
        [/initials]

        [savers]
            [saver]
                name = RectGridPointSaver
                path = ../../seismograms/seismogram_{num1}/seismogram_{num1}_{num2}/seismogram.txt
                params = vx, vy
                order = 1
                save = 1
                start = 16, -100
                step = 16, 0.0
                num = 312
                norms = 0, 0
            [/saver]
        [/savers]
        """
    with open(f"./dataset/configs/config_{num1}/config_{num1}_{num2}.conf", "w") as f:
        f.write(config)


if __name__ == "__main__":
    for i in range(1):
        model_generator(i)
        for j in range(3):
            config_generator(i, j)
