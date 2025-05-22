class Constants:
    # параметры модели и данных
    model_x_size, model_y_size = 501, 201  # Размеры модели (длина и глубина)
    seism_x_size, seism_y_size = 624, 500  # Размеры сейсмограмм
    # в txt файлам размер по X 625 - добавляется столбик времени, который мы выкидываем

    seism_ch_size = (500, 312)
    # приходится делать sample_shape немного больше seism_ch_size, чтобы они были кратны 2^n
    sample_shape = (512, 384)
    # уменьшенный размер sample_shape если нужно экономить время
    # sample_shape = (256, 256) # демо размер семпла

    # физические параметры для нормализации геомоделей:
    delimiter_range = [2.0, 3.0]
    multiplicator_range = [1.1, 1.15]

    rho_range = [2000.0, 5000.0]
    vp_range = [rho_range[0] * multiplicator_range[0], rho_range[1] * multiplicator_range[1]]  # км/с
    vs_range = [vp_range[0] / delimiter_range[1], vp_range[1] / delimiter_range[0]]
