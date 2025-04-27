class CFG:
    # параметры модели и данных
    model_x_size, model_y_size = 501, 201  # Размеры модели (длина и глубина)
    seism_x_size, seism_y_size = 625, 500  # Размеры сейсмограмм

    # физические параметры для нормализации геомоделей:
    delimiter_range = [2.0, 3.0]
    multiplicator_range = [1.1, 1.15]

    rho_range = [2000.0, 5000.0]
    vp_range = [rho_range[0] * multiplicator_range[0], rho_range[1] * multiplicator_range[1]]  # км/с
    vs_range = [vp_range[0] / delimiter_range[1], vp_range[1] / delimiter_range[0]]

    # параметры трейна
    USE_MULTIGPU = True
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-3
    EPOCHS = 100
