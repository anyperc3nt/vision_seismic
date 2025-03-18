import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
import math
import os


def model_generator(num):
    def ragged_line(x):
        return ((math.sin((x / 10  * (math.pi / 4))) + math.cos((x / np.random.uniform(10.0, 12.0)  * (math.pi / 4)))) / 2) * np.random.uniform(2.0, 3.0)

    if not os.path.exists(f'./dataset/models/model_{num}'):
        os.makedirs(f'./dataset/models/model_{num}', exist_ok=True)

    if not os.path.exists(f'./dataset/configs/config_{num}'):
        os.makedirs(f'./dataset/configs/config_{num}', exist_ok=True)

    # Параметры модели
    x_size, y_size = 501, 201  # Размеры модели (длина и глубина)
    H = 200  # Глубина в метрах
    num_layers = np.random.randint(3, 11)  # Случайное количество слоев

    # Диапазоны параметров слоев
    rho_min, rho_max = 2000.0, 5000.0 
    vp_min, vp_max = 2200.0, 5000.0  # км/с
    delimiter = np.random.uniform(2.0, 3.0)
    multiplicator = np.random.uniform(1.1, 1.15)

    # Генерация параметров слоев
    layer_thicknesses = np.random.uniform(0.7 * H / num_layers, 1.4 * H / num_layers, num_layers)
    layer_depths = np.cumsum(layer_thicknesses).astype(int)
    layer_depths[layer_depths >= y_size] = y_size - 1  # Ограничение по глубине

    rho = [np.random.uniform(rho_min, rho_min+1.5)]
    vp = [rho[0] * multiplicator]
    vs = [vp[0] / delimiter]

    for i in range(1, num_layers):
        temp_rho = rho[i-1] * np.random.uniform(1, 2.4 ** (1 / num_layers))
        rho.append(temp_rho if temp_rho < rho_max else rho_max)
        temp_vp = rho[i] * multiplicator
        vp.append(temp_vp if temp_vp < vp_max else vp_max)
        vs.append(vp[i] / delimiter)

    # Создание 2D модели
    model = np.zeros((x_size, y_size))
    anomaly_type = np.random.choice(["ellipse", "mountain", "distorted_layers", "null"])
    print(anomaly_type)

    for i in range(num_layers):
        start = layer_depths[i - 1] if i > 0 else 0
        # end = layer_depths[i]

        for x in range(x_size):     
            if start == 0:
                model[x, :] = i + 1
            # elif num_layers - 1 == i:
            #     model[x, int(start + ragged_line(x)):] = i + 1
            else:      
                 model[x, int(start + ragged_line(x)):] = i + 1

    # Добавление аномалий
    if anomaly_type == "ellipse":
        cx, cy = np.random.randint(100, 350), np.random.randint(20, 150) 
        rx, ry = np.random.randint(5, 50), np.random.randint(6, 20) 

        for x in range(x_size):
            for y in range(y_size):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                    model[x, y] = num_layers + 1

    elif anomaly_type == "mountain":
        x_points = np.linspace(200, 300, num=4)
        y_points = np.random.uniform(0, 50, size=4)
        y_points[0], y_points[-1] = 0, 0
        spline = CubicSpline(x_points, y_points)

        for x in range(x_size):
            y_top = int(spline(x)) if 200 <= x < 300 else 0
            model[x, int(y_size - y_top):] = num_layers + 1

    elif anomaly_type == "distorted_layers":
        x_points = np.linspace(0, 500, num=3)
        y_points = np.random.uniform(5, 100, size=3)
        y_points[0], y_points[-1] = 0, 150
        spline = CubicSpline(x_points, y_points)

        for x in range(x_size):
            y_top = int(spline(x) + x / 50) if x >= x_points[0] else 0
            model[x, int((y_size - y_top) + 3 * ragged_line(x + 2)):] = num_layers + 1
    else:
        pass

    # Визуализация распределений rho, vp и vs
    def save_distribution(data, title, filename):
        plt.figure(figsize=(8, 6))
        plt.imshow(data.T, cmap='viridis', norm='log')
        plt.axis('off')  # Убираем оси
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    # Распределение rho
    rho_model = np.zeros_like(model, dtype=float)
    for i in range(num_layers + 1):
        if i < len(rho):
            rho_model[model == i + 1] = rho[i]
        else:
            rho_model[model == i + 1] = 4800.0
    print(num_layers)
    rho_model_s = np.flip(rho_model.T, axis=0)
    rho_model_s.astype('f').tofile(f'./dataset/configs/config_{num}/rho_{num}.bin')
    save_distribution(rho_model, 'rho', f'./dataset/models/model_{num}/rho.png')

    # Распределение vp
    vp_model = np.zeros_like(model, dtype=float)
    for i in range(num_layers + 1):
        if i < len(vp):
            vp_model[model == i + 1] = vp[i]
        else:
            vp_model[model == i + 1] = 4600.0
    vp_model_s = np.flip(vp_model.T, axis=0)
    vp_model_s.astype('f').tofile(f'./dataset/configs/config_{num}/vp_{num}.bin')
    save_distribution(vp_model, 'Vp', f'./dataset/models/model_{num}/vp.png')

    # Распределение vs
    vs_model = np.zeros_like(model, dtype=float)
    for i in range(num_layers + 1):
        if i < len(vs):
            vs_model[model == i + 1] = vs[i]
        else:
            vs_model[model == i + 1] = 2000.0 + ((vs[-1] - np.average(vs)) / (vp_max / delimiter))
    vs_model_s = np.flip(vs_model.T, axis=0)
    vs_model_s.astype('f').tofile(f'./dataset/configs/config_{num}/vs_{num}.bin')
    save_distribution(vs_model, 'Vs', f'./dataset/models/model_{num}/vs.png')

def config_generator(num1, num2):
    if not os.path.exists(f'./dataset/seismograms/seismogram_{num1}'):
        os.makedirs(f'./dataset/seismograms/seismogram_{num1}', exist_ok=True)

    if not os.path.exists(f'./dataset/seismograms/seismogram_{num1}/seismogram_{num1}_{num2}'):
        os.makedirs(f'./dataset/seismograms/seismogram_{num1}/seismogram_{num1}_{num2}', exist_ok=True)

    x_coord = 100 + 2400 * num2
    config = f'''

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
                                file = ../../../ricker_30.txt
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
        '''
    with open(f'./dataset/configs/config_{num1}/config_{num1}_{num2}.conf', 'w') as f:
            f.write(config)


if __name__ == "__main__":
    for i in range(1000):
        model_generator(i)
        for j in range(3):
            config_generator(i, j)
