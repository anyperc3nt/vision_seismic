from typing import NamedTuple

# типы подходов

"""    
    "vs"  # используем только компоненту vs -> 3 канала
    "vp"  # используем только компоненту vp -> 3 канала
    "3ch"  # старый подход: все вперемешку, в каждом канале поочередно столбиками идут vp и vs
    "6ch"  # подаем и vs и vp -> 6 каналов
"""


class CFG:
    # параметры модели и данных
    model_x_size, model_y_size = 501, 201  # Размеры модели (длина и глубина)
    seism_x_size, seism_y_size = 624, 500  # Размеры сейсмограмм
    # в txt файлам размер по X 625 - добавляется столбик времени, который мы выкидываем

    # физические параметры для нормализации геомоделей:
    delimiter_range = [2.0, 3.0]
    multiplicator_range = [1.1, 1.15]

    rho_range = [2000.0, 5000.0]
    vp_range = [rho_range[0] * multiplicator_range[0], rho_range[1] * multiplicator_range[1]]  # км/с
    vs_range = [vp_range[0] / delimiter_range[1], vp_range[1] / delimiter_range[0]]

    # Переменные отвечающие за тип подхода
    CHANNEL_TAG = "6ch"
    if CHANNEL_TAG in ("vs", "vp", "6ch"):
        CHANNEL_DELIMITER = 2
    elif CHANNEL_TAG == "3ch":
        CHANNEL_DELIMITER = 1

    # параметры трейна
    USE_MULTIGPU = True
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-2
    EPOCHS = 10
