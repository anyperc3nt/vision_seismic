import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from CFG import CFG
from data_preprocessing import normalize_geomodel, normalize_seism
from torch.utils.data import DataLoader, Dataset


def load_seism(seism_paths, index):

    seism_0 = np.loadtxt((seism_paths[index][0]), dtype="f")[:,1:]
    seism_1 = np.loadtxt((seism_paths[index][1]), dtype="f")[:,1:]
    seism_2 = np.loadtxt((seism_paths[index][2]), dtype="f")[:,1:]
    seism_3ch = np.stack((seism_0, seism_1, seism_2), axis=-1)

    return seism_3ch


def load_geomodel(model_paths, index):

    model_0 = np.fromfile((model_paths[index][0]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)
    model_1 = np.fromfile((model_paths[index][1]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)
    model_2 = np.fromfile((model_paths[index][2]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)
    geomodel_3ch = np.stack((model_0, model_1, model_2), axis=-1)

    return geomodel_3ch


def load_fault(fault_paths, index):

    fault = np.fromfile((fault_paths[index]), dtype="f").reshape(CFG.model_y_size, CFG.model_x_size)

    return fault


base_transform = A.Compose(
    [
        A.Resize(height=256, width=640, interpolation=cv2.INTER_CUBIC),
        ToTensorV2(),
    ],
is_check_shapes=False,
additional_targets={'seism': 'image'},
)


class Seism_3ch_Dataset(Dataset):
    def __init__(
        self,
        seism_paths,
        geomodel_paths,
        transform=base_transform,
    ):
        self.seism_paths = seism_paths
        self.geomodel_paths = geomodel_paths
        self.transform = transform

    def __len__(self):
        return len(self.seism_paths)

    def __getitem__(self, index):

        seism_3ch = load_seism(self.seism_paths, index)
        geomodel_3ch = load_geomodel(self.geomodel_paths, index)

        seism_3ch = normalize_seism(seism_3ch)
        geomodel_3ch = normalize_geomodel(geomodel_3ch)

        if self.transform is not None:
            augmented = self.transform(image=geomodel_3ch, seism=seism_3ch)
            geomodel_3ch = augmented['image']
            seism_3ch = augmented['seism']

        return seism_3ch, geomodel_3ch
