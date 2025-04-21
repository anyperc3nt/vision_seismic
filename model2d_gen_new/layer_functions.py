import numpy as np
from typing import Tuple
from perlin_noise import PerlinNoise


def ragged_line(x):
    return (
        (np.sin((x / 10 * (np.pi / 4))) + np.cos((x / np.random.uniform(10.0, 12.0) * (np.pi / 4)))) / 2
    ) * np.random.uniform(2.0, 3.0)


# Немного стремная, но оптимизация perlin_noise, в 200*num_samples раз сокращающее время на повторные рассчеты
noise1 = PerlinNoise(octaves=3)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=12)

# Предполагаем, что ширина модели не превышает MAX_X
MAX_X = 2000

# Один раз строим таблицу
_perlin_table = np.empty(MAX_X, dtype=float)
for xi in range(MAX_X):
    _perlin_table[xi] = noise1(xi / 500) * 40 + noise2(xi / 500) * 20 + noise3(xi / 500) * 10


def perlin_line(x):
    """
    x  – либо одно целое, либо массив целых.
    Возвращает предвычисленные значения из таблицы.
    """
    arr = np.asarray(x, dtype=int)
    # Если вдруг x выходит за пределы [0, MAX_X), обрежем
    arr_clipped = np.clip(arr, 0, MAX_X - 1)
    return _perlin_table[arr_clipped]


def zero_line(x):
    return 0
