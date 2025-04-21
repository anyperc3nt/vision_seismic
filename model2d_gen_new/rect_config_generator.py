import os

def rect_config_generator(dataset_path,num1, num2):
    if not os.path.exists(f'{dataset_path}/seismograms/seismogram_{num1}'):
        os.makedirs(f'{dataset_path}/seismograms/seismogram_{num1}', exist_ok=True)

    if not os.path.exists(f'{dataset_path}/seismograms/seismogram_{num1}/seismogram_{num1}_{num2}'):
        os.makedirs(f'{dataset_path}/seismograms/seismogram_{num1}/seismogram_{num1}_{num2}', exist_ok=True)

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
                                file = ../../../../ricker_30.txt
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
                save = 4
                start = 16, -100
                step = 16, 0.0
                num = 312
                norms = 0, 0
            [/saver]
        [/savers]
        '''
    with open(f'{dataset_path}/configs/config_{num1}/config_{num1}_{num2}.conf', 'w') as f:
            f.write(config)
