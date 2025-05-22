import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from callbacks import *
from data_pipeline.data_preprocessing import NpyBuilder
from data_pipeline.path_utils import generate_lists_split, load_fault_paths, load_model_paths, load_seism_paths
from data_pipeline.torch_dataloader import SeismicDataModule
from fault_losses_and_metrics import *
from geo_losses import *
from models.Unet_model_girafe import UNet_girafe
from models.Unet_R2SE import UNet_R2SE
from pytorch_lightning.loggers import CSVLogger

# os.environ["CLEARML_CONFIG_FILE"] = "<path to your .clearml.conf>"
# from clearml import Logger, Task


class SeismicLightningModel(pl.LightningModule):
    def __init__(self, model, lr, criterion, metrics: list):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = criterion
        self.metrics = nn.ModuleList(metrics)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        for metric in self.metrics:
            score = metric(y_hat, y)
            name = metric.__class__.__name__
            self.log(f"val_metric_{name}", score, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


class CFG:
    # постановка задачи
    """
    "geomodel" -> out channels: 3x geomodel (аномалии)
    "geomodel+faults" -> out channels: 3x geomodel, 1x faults map
    "faults" -> 1x faults map
    """
    faults = False
    geomodels = True
    out_channels = 3 * geomodels + 1 * faults

    # Переменные отвечающие за тип подхода
    """
    "vp"  # используем только компоненту vp -> 3 канала
    "vs"  # используем только компоненту vs -> 3 канала
    "vp+vs"  # подаем и vs и vp -> 6 каналов
    """
    channel_tag = "vp"

    # параметры трейна
    lr = 1e-2 * 0.3
    num_epochs = 50
    batch_size = 4

    """логика, вычисляющая параметры для модели"""
    in_channel_multiplier = 1
    if channel_tag == "vp+vs":
        in_channel_multiplier = 2  # в 2 раза больше каналов подается на вход модели


def main():
    torch.set_float32_matmul_precision("medium")  # Оптимизация matmul для Tensor Cores

    """подготовка датасета"""
    # BASE_PATH = "/home/nik/dataset_faults/faults_sus"
    BASE_PATH = "/home/nik/dataset_04_24/anomaly_hard"

    seism_paths = load_seism_paths(BASE_PATH)
    model_paths = load_model_paths(BASE_PATH)
    fault_paths = None
    if CFG.faults:
        fault_paths = load_fault_paths(BASE_PATH)

    cache_dataset = NpyBuilder(BASE_PATH, seism_paths, model_paths, fault_paths, force=False)

    train_list, val_list = generate_lists_split(BASE_PATH)
    data_module = SeismicDataModule(
        train_list=train_list,
        val_list=val_list,
        channel_tag=CFG.channel_tag,
        out_channels=CFG.out_channels,
        geomodels=CFG.geomodels,
        batch_size=CFG.batch_size,
        num_workers=4,
    )
    """инициализация модели, лосса, метрик"""
    # model = UNet_girafe(in_channels=3 * CFG.in_channel_multiplier, num_classes=CFG.out_channels)
    model = UNet_R2SE(in_channels=3 * CFG.in_channel_multiplier, num_classes=CFG.out_channels)

    criterion = CombinedMSETverskyLoss(mse_weight=0.15, tversky_alpha=0.9, tversky_beta=0.1)
    criterion = nn.MSELoss()

    # faults
    # lightning_model = SeismicLightningModel(
    #     model, lr=CFG.lr, criterion=criterion, metrics=[SSIM3Channels(), FaultIoU(channel=3)]
    # )
    # anomalies
    lightning_model = SeismicLightningModel(model, lr=CFG.lr, criterion=criterion, metrics=[SSIM3Channels()])

    """clearml"""
    # task = Task.init(
    #     project_name='vision-seismic',
    #     task_name='testing pytorch lightning',
    #     tags=['Unet', 'Faults']
    # )
    """инициализация трейна"""
    model_name = model.__class__.__name__
    loss_name = criterion.__class__.__name__
    now = datetime.now().strftime("%m%d_%H%M%S")
    dir_name = f"{now}_{model_name}_{loss_name}_epoch{CFG.num_epochs}"
    logger = CSVLogger(save_dir="lightning_logs", name=dir_name)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            GlobalEpochProgressBar(),
            RenameLogDirOnExceptionCallback(min_epoch=15),
            PlottingCallback(silent=False),
            SamplePredictionCallback(silent=False),
            DictLoggerCallback(CFG),
            DictLoggerCallback({"BASE_PATH": BASE_PATH}),
            LogReprCallback({"loss": criterion}),
        ],
        num_sanity_val_steps=0,
        max_epochs=CFG.num_epochs,
        devices=[0, 1],
        strategy="ddp",
        # accelerator="gpu",
        # devices=1,
        precision="16-mixed",
        # precision='32-true',
        enable_progress_bar=False,
    )

    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    main()
