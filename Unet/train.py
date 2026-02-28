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

# from models.UNetR2SE_AGSelf import UNetR2SE_AGSelf
# from models.UNetR2SE_stride import UNetR2SE_stride
# from models.UNetR2SE_ContextSkip import UNetR2SE_ContextSkip
# from models.UNetR2SE_DeformableSkip import UNetR2SE_DeformableSkip
# from models.UNetR2SE_new import UNetR2SE
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import CSVLogger


class SeismicLightningModel(pl.LightningModule):
    def __init__(self, model, lr, criterion, metrics: list):
        super().__init__()
        self.lr = lr
        self.criterion = criterion
        self.metrics = nn.ModuleList(metrics)
        self.model = model

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
    faults = True
    geomodels = True
    out_channels = 3 * geomodels + 1 * faults

    # fault_distance = 6
    fault_distance = 3

    # Переменные отвечающие за тип подхода
    """
    "vp"  # используем только компоненту vp -> 3 канала
    "vs"  # используем только компоненту vs -> 3 канала
    "vp+vs"  # подаем и vs и vp -> 6 каналов
    """
    channel_tag = "vp+vs"

    # параметры трейна
    lr = 1e-3
    num_epochs = 65
    # num_epochs = 7
    batch_size = 4

    """логика, вычисляющая параметры для модели"""
    in_channel_multiplier = 1
    if channel_tag == "vp+vs":
        in_channel_multiplier = 2  # в 2 раза больше каналов подается на вход модели


def main():
    torch.set_float32_matmul_precision("medium")  # Оптимизация matmul для Tensor Cores

    """подготовка датасета"""
    # BASE_PATH = "/home/nik/dataset_faults/512/faults_a_v3"
    # BASE_PATH = "/home/nik/dataset_faults/512/faults_b_v2"
    BASE_PATH = "/home/nik/dataset_faults/512/faults_c"

    seism_paths = load_seism_paths(BASE_PATH)
    model_paths = load_model_paths(BASE_PATH)
    fault_paths = None
    if CFG.faults:
        fault_paths = load_fault_paths(BASE_PATH)

    _cache_dataset = NpyBuilder(BASE_PATH, seism_paths, model_paths, fault_paths, force=False)

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
    model = UNetR2SE(in_channels=3 * CFG.in_channel_multiplier, num_classes=CFG.out_channels, faults=CFG.faults)

    criterion = LossCombinator(
        losses=[nn.MSELoss(), FocalLoss(alpha=0.75, gamma=2)],
        channels=[[0, 3], [3, 4]],
        weights=[0.5, 0.5],
        flatten_channels=[False, True],
    )

    lightning_model = SeismicLightningModel(
        model,
        lr=CFG.lr,
        criterion=criterion,
        metrics=[SSIM3Channels(), FaultIoU(channel=3)],
    )

    """инициализация трейна"""
    comment = "тест_С"

    model_name = model.__class__.__name__
    loss_name = str(criterion)
    dataset_name = os.path.basename(BASE_PATH)
    now = datetime.now().strftime("%Y-%m-%d_at_%H-%M-%S")
    dir_name = f"{now}_{comment}_ep{CFG.num_epochs}"
    logger = CSVLogger(save_dir=f"lightning_logs/{dataset_name}/{loss_name}", name=model_name, version=dir_name)

    checkpoint_best_iou = ModelCheckpoint(
        monitor="val_metric_FaultIoU",
        mode="max",
        save_top_k=1,
        filename="best_iou-epoch={epoch:02d}-iou={val_metric_FaultIoU:.4f}",
        dirpath=logger.log_dir,
        auto_insert_metric_name=False,
    )

    # 2. Колбек для сохранения модели с наименьшим val_loss
    checkpoint_best_loss = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_loss-epoch={epoch:02d}-loss={val_loss:.4f}",
        dirpath=logger.log_dir,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        logger=logger, 
        callbacks=[
            checkpoint_best_iou,
            checkpoint_best_loss,
            GlobalEpochProgressBar(),
            RenameLogDirOnExceptionCallback(dir_name, min_epoch=10),
            PlottingCallback(plot_name=f"{dataset_name} {model_name}", silent=True),
            # SamplePredictionCallback(n_val=5, n_final=15, silent=False),
            # PRPlotCallback(channel=3),
            DictLoggerCallback(CFG),
            DictLoggerCallback({"BASE_PATH": BASE_PATH}),
            DictLoggerCallback({"model_class_name": model.__class__.__name__}),
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
