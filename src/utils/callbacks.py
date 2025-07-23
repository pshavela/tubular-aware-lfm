import torch
import lightning as L
from functools import partial
from lightning.pytorch.callbacks.lambda_function import LambdaCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def _gpu_stats_callback(
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
        outputs = None,
        suffix: str = '',
        additional_devices = None
    ):
    devices = [pl_module.device]
    if additional_devices:
        for device in additional_devices:
            devices.append(device)

    for device in devices:
        free, total = torch.cuda.mem_get_info(device)
        vram_total_gb = total / (1024 ** 3)
        vram_used_gb = (total - free) / (1024 ** 3)
        pl_module.log_dict({
            f'gpu/{device}_vram_total_gb' : vram_total_gb,
            f'gpu/{device}_vram_used_gb.{suffix}' : vram_used_gb,
            f'gpu/{device}_vram_free_gb.{suffix}' : vram_total_gb - vram_used_gb,
        })


def GPUStatsCallback(on_val: bool = False, additional_devices = None):
    on_validation_batch_start = partial(_gpu_stats_callback, suffix='batch_val_start',
                                        additional_devices=additional_devices),
    on_validation_batch_end = partial(_gpu_stats_callback, suffix='batch_val_end',
                                      additional_devices=additional_devices)
    on_train_batch_start = partial(_gpu_stats_callback, suffix='batch_train_start',
                                   additional_devices=additional_devices)
    on_train_batch_end = partial(_gpu_stats_callback, suffix='batch_train_end',
                                 additional_devices=additional_devices)

    if on_val:
        return LambdaCallback(
            on_validation_batch_start=on_validation_batch_start,
            on_validation_batch_end=on_validation_batch_end,
            on_train_batch_start=on_train_batch_start,
            on_train_batch_end=on_train_batch_end
        )

    return LambdaCallback(
            on_train_batch_start=on_train_batch_start,
            on_train_batch_end=on_train_batch_end
    )


class ResumableEarlyStopping(EarlyStopping):
    """
    Small hack to prevent early stopping callback from stopping after one additional epoch when resuming trainer's fit.
    """
    def load_state_dict(self, state_dict: dict) -> None:
        self.best_score = state_dict["best_score"]
