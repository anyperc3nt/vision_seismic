from typing import Tuple

import numpy as np
from perlin_noise import PerlinNoise


def ragged_line(x):
    return (
        (np.sin((x / 10 * (np.pi / 4))) + np.cos((x / np.random.uniform(10.0, 12.0) * (np.pi / 4)))) / 2
    ) * np.random.uniform(2.0, 3.0)


# Немного стремная, но оптимизация perlin_noise, в 500*num_samples раз сокращающее время на повторные рассчеты
MAX_X = 2000


class PerlinLine:
    """
    PerlinLine создавалась как "сложная" функция слоев для обучения нейросети,
    поэтому справедливым будет, если для каждого семпла она будет разная
    (до этого она инициализировалась 1 раз при создании датасета, т. е. была идентичной для всех семплов)
    """

    def __init__(self):
        noise1 = PerlinNoise(octaves=12)
        noise2 = PerlinNoise(octaves=24)
        noise3 = PerlinNoise(octaves=48)

        self._table1 = np.array([noise1(x / MAX_X) * 40 * 1.5 for x in range(MAX_X)])
        self._table2 = np.array([noise2(x / MAX_X) * 20 * 1.5 for x in range(MAX_X)])
        self._table3 = np.array([noise3(x / MAX_X) * 10 * 1.5 for x in range(MAX_X)])

        self.shift1 = 0
        self.shift2 = 0
        self.shift3 = 0

    def __call__(self, x):
        x = np.asarray(x, dtype=int) % MAX_X
        return (
            self._table1[(x + self.shift1) % MAX_X]
            + self._table2[(x + self.shift2) % MAX_X]
            + self._table3[(x + self.shift3) % MAX_X]
        )

    def shuffle(self):
        """
        меняет сдвиги для трех шумов, из которых состоит функция слоя, делая ее неузнаваемой
        """
        self.shift1 = np.random.randint(0, MAX_X)
        self.shift2 = np.random.randint(0, MAX_X)
        self.shift3 = np.random.randint(0, MAX_X)


def zero_line(x):
    return 0
