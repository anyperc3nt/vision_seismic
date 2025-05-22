import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SeismicDataset(Dataset):
    def __init__(self, file_paths, channel_tag, out_channels, geomodels):
        self.file_paths = file_paths
        self.channel_tag = channel_tag
        self.out_channels = out_channels
        self.geomodels = geomodels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], mmap_mode="r")  # mmap_mode="r" - оптимизация подгрузки файла
        # Объединяем каналы
        if self.channel_tag == "vp+vs":
            seism = data[:, :, 0:6]
        elif self.channel_tag == "vp":
            seism = data[:, :, 0:3]
        elif self.channel_tag == "vs":
            seism = data[:, :, 3:6]

        if self.geomodels:
            geo = data[:, :, 6 : 6 + self.out_channels]
        else:
            geo = data[:, :, 9 : 9 + self.out_channels]

        seism = torch.tensor(seism.transpose(2, 0, 1))
        geo = torch.tensor(geo.transpose(2, 0, 1))

        return seism, geo


class SeismicDataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, channel_tag, out_channels, geomodels, batch_size, num_workers):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.channel_tag = channel_tag
        self.out_channels = out_channels
        self.geomodels = geomodels

    def train_dataloader(self):
        train_dataset = SeismicDataset(self.train_list, self.channel_tag, self.out_channels, self.geomodels)
        sampler = DistributedSampler(train_dataset, shuffle=True)
        return DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        val_dataset = SeismicDataset(self.val_list, self.channel_tag, self.out_channels, self.geomodels)
        sampler = DistributedSampler(val_dataset, shuffle=False)
        return DataLoader(
            val_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
