import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from tqdm.auto import tqdm
from visualization.visualizers import make_losses_and_metrics_fig, make_samples_fig


class GlobalEpochProgressBar(Callback):
    def on_train_start(self, trainer, pl_module):
        self.epochs = trainer.max_epochs
        self.pbar = tqdm(
            total=self.epochs,
            desc="Epochs",
            unit="epoch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.pbar.update(1)
        elapsed = time.time() - self.start_time
        avg_epoch_time = elapsed / (trainer.current_epoch + 1)
        remaining = avg_epoch_time * (self.epochs - trainer.current_epoch - 1)
        self.pbar.set_postfix(elapsed=f"{int(elapsed)}s", remaining=f"{int(remaining)}s")
        self.pbar.refresh()  # гарантирует обновление вывода

    def on_train_end(self, trainer, pl_module):
        self.pbar.close()


class RenameLogDirOnExceptionCallback(pl.Callback):
    """
    переименовывает директорию в случае прерывания трейна
    (удобно для отсеивания незавершенных экспериментов)
    <имя>epoch_MAX -> <имя>epoch_на_которой_прервали

    min_epoch: в случае, если прошло меньше min_epoch эпох, папка с логами вообще удаляется
    """

    def __init__(self, min_epoch=10):
        self.min_epoch = min_epoch

    def on_exception(self, trainer, pl_module, exception):
        rank = pl_module.global_rank
        if rank != 0:
            return
        if trainer.current_epoch >= self.min_epoch:
            self.rename_on_exit(trainer, pl_module, trainer.current_epoch)
        else:
            shutil.rmtree(os.path.dirname(trainer.logger.log_dir))
            print(
                f"RenameLogDirOnExceptionCallback: Прошло меньше min_epoch={self.min_epoch}, логи удалены, чтобы не засорять директорию"
            )

    def rename_on_exit(self, trainer, pl_module, epoch):
        old_log_dir = trainer.logger.log_dir
        if not os.path.exists(old_log_dir):
            return

        old_base_log_dir = os.path.dirname(old_log_dir)
        base_dir = os.path.dirname(old_base_log_dir)
        model_name = pl_module.model.__class__.__name__ if hasattr(pl_module, "model") else "UnknownModel"

        criterion = getattr(pl_module, "criterion", None)
        loss_name = criterion.__class__.__name__ if criterion else "UnknownLoss"

        now = datetime.now().strftime("%m%d_%H%M%S")
        new_dir_name = f"{now}_{model_name}_{loss_name}_epoch{epoch}_aborted"
        new_log_dir = os.path.join(base_dir, new_dir_name)

        try:
            shutil.move(old_log_dir, new_log_dir)
            shutil.rmtree(old_base_log_dir)
            print(f"RenameLogDirOnExceptionCallback: Логи перемещены в: {new_log_dir}")
        except Exception as e:
            print(f"RenameLogDirOnExceptionCallback: Не удалось переименовать лог-папку: {e}")


class PlottingCallback(Callback):
    def __init__(self, silent=False):
        self.silent = silent
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = {}

    # def on_exception(self, trainer, pl_module, exception):
    #     self.on_fit_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss_epoch")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        rank = pl_module.global_rank
        if rank != 0:
            return
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.item())
        for name, value in trainer.callback_metrics.items():
            if name.startswith("val_metric_") and value is not None:
                if name not in self.val_metrics_history:
                    self.val_metrics_history[name] = []
                self.val_metrics_history[name].append(value.item())
        if not self.silent:
            self.plot()
            log_dir = trainer.logger.log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.plot(os.path.join(log_dir, "loss_plot.png"))

    def on_fit_end(self, trainer, pl_module):
        rank = pl_module.global_rank
        if rank != 0:
            return
        if trainer.logger is not None:
            log_dir = trainer.logger.log_dir
            os.makedirs(log_dir, exist_ok=True)
            save_path = os.path.join(log_dir, "loss_plot_final.png")
        else:
            save_path = "img_logs/loss_plot_final.png"
        self.plot(save_path)

    def plot(self, path="img_logs/loss_plot.png"):
        metrics = list(self.val_metrics_history.values())
        metrics_labels = list(self.val_metrics_history.keys())
        fig = make_losses_and_metrics_fig(
            losses=[self.train_losses, self.val_losses],
            loss_labels=["train loss", "val loss"],
            metrics=metrics,
            metrics_labels=metrics_labels,
            title=None,
            xlabel="Epoch",
            ylabel="loss(logscale)",
            logscale=True,
        )
        fig.savefig(path)
        plt.close(fig)


class SamplePredictionCallback(Callback):
    def __init__(self, silent=False, final_batches=3):
        self.logged = False  # Чтобы логировать только один раз за валидацию
        self.silent = silent
        self.final_batches = final_batches

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.logged:
            return

        rank = pl_module.global_rank
        if rank != 0 or batch_idx != 0:
            return

        if self.silent:
            return

        x, y = batch
        y_hat = pl_module(x)

        fig = make_samples_fig(
            seisms=x[0:3],
            geo_predicts=y_hat[0:3],
            geo_targets=y[0:3],
        )
        save_path = "img_logs/val_samples.png"
        fig.savefig(save_path)
        log_dir = trainer.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        fig.savefig(os.path.join(log_dir, "val_samples.png"))
        plt.close(fig)
        self.logged = True

    def on_validation_epoch_end(self, trainer, pl_module):
        self.logged = False  # Сбросить флаг на следующую эпоху

    def on_fit_end(self, trainer, pl_module):
        if pl_module.global_rank != 0:
            return

        val_loader = None
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            val_loader = trainer.datamodule.val_dataloader()
        elif trainer.val_dataloaders:
            val_loader = trainer.val_dataloaders[0]
        else:
            return

        seisms_all, predicts_all, targets_all = [], [], []
        for i, batch in enumerate(val_loader):
            if i >= self.final_batches:
                break
            x, y = batch
            y_hat = pl_module(x)

            seisms_all.append(x)
            predicts_all.append(y_hat)
            targets_all.append(y)

        import torch

        seisms_all = torch.cat(seisms_all, dim=0)
        predicts_all = torch.cat(predicts_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)

        if trainer.logger is not None:
            log_dir = trainer.logger.log_dir
            os.makedirs(log_dir, exist_ok=True)
            save_path = os.path.join(log_dir, "final_samples_combined.png")
        else:
            save_path = "img_logs/loss_plot_final.png"

        fig = make_samples_fig(
            seisms=seisms_all,
            geo_predicts=predicts_all,
            geo_targets=targets_all,
        )
        fig.savefig(save_path)
        plt.close(fig)


class DictLoggerCallback(Callback):
    def __init__(self, cfg_class):
        self.cfg_class = cfg_class

    def on_fit_start(self, trainer, pl_module):
        if isinstance(self.cfg_class, dict):
            cfg_dict = self.cfg_class
        else:
            cfg_dict = {k: v for k, v in self.cfg_class.__dict__.items() if not k.startswith("__") and not callable(v)}
        trainer.logger.log_hyperparams(cfg_dict)


class LogReprCallback(Callback):
    def __init__(self, var, name="custom_object"):
        self.var = var
        self.name = name

    def on_fit_start(self, trainer, pl_module):
        trainer.logger.log_hyperparams({self.name: repr(self.var)})
