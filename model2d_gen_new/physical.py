import numpy as np

from model import GeoModel

class PhysicalModelBuilder:
    def __init__(self, rho_range, vp_range, delimiter_range, multiplicator_range):
        self.rho_min, self.rho_max = rho_range[0], rho_range[1]
        self.vp_min, self.vp_max = vp_range[0],  vp_range[1]
        self.delimiter_range = delimiter_range
        self.multiplicator_range = multiplicator_range

    def generate_properties(self, num_layers: int):

        self.delimiter = np.random.uniform(*self.delimiter_range)
        self.multiplicator = np.random.uniform(*self.multiplicator_range)

        rho = [np.random.uniform(self.rho_min, self.rho_min + 1.5)]
        vp = [rho[0] * self.multiplicator]
        vs = [vp[0] / self.delimiter]

        for i in range(1, num_layers):
            temp_rho = rho[i - 1] * np.random.uniform(1, 2.4 ** (1 / num_layers))
            rho.append(min(temp_rho, self.rho_max))

            temp_vp = rho[i] * self.multiplicator
            vp.append(min(temp_vp, self.vp_max))

            vs.append(vp[i] / self.delimiter)

        return rho, vp, vs

    def build_maps(self, model: np.ndarray, num_layers):
        rho_list, vp_list, vs_list = self.generate_properties(num_layers)

        rho_model = np.zeros_like(model, dtype=float)
        vp_model = np.zeros_like(model, dtype=float)
        vs_model = np.zeros_like(model, dtype=float)


        for i in range(num_layers + 1):
            idx = i + 1
            if i < len(rho_list):
                rho_model[model == idx] = rho_list[i]
                vp_model[model == idx] = vp_list[i]
                vs_model[model == idx] = vs_list[i]
            else:
                #магические числа, которые пока нельзя задать конфигом
                rho_model[model == idx] = 4800.0
                vp_model[model == idx] = 4600.0
                vs_model[model == idx] = 2000.0 + ((vs_list[-1] - np.mean(vs_list)) / (self.vp_max / self.delimiter))

        return rho_model, vp_model, vs_model