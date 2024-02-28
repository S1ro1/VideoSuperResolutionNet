from typing import Any
import torch
import lightning as L
from torch import nn, optim
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import cv2

torch.set_float32_matmul_precision('medium')


class VideoSRLightningModule(L.LightningModule):
    def __init__(self, args: dict[str, Any], num_frames: int = 1):
        super().__init__()
        self.num_frames = num_frames
        self.args = args
        self._setup()
        self.save_hyperparameters()

    def _setup(self) -> None:
        self.lr = self.args["lr"]

        if self.args["model_name"] == "srresnet":
            from model_zoo.srresnet import srresnet_x4 as Model
        elif self.args["model_name"] == "EDVR":
            from basicsr.archs.edvr_arch import EDVR as Model
        else:
            raise NotImplementedError(
                f"Model {self.args['model_name']} has not been implemented yet.")

        self.model = Model(**self.args["model_args"])

    def _compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, compute_loss: bool = True) -> dict[str, torch.Tensor]:
        metrics = {}
        print(f"{outputs.shape=}, {targets.shape=}")
        metrics["ssim"] = structural_similarity_index_measure(
            outputs, targets)
        metrics["psnr"] = peak_signal_noise_ratio(outputs, targets)
        if compute_loss:
            metrics["loss"] = nn.functional.mse_loss(outputs, targets)
        return metrics

    def _log_metrics(self, metrics: dict[str, torch.Tensor], prefix: str, sync_dist: bool = False) -> None:
        for key, value in metrics.items():
            self.log(f"{prefix}/{key}", value, prog_bar=True,
                     logger=True, sync_dist=sync_dist)

    def _log_images(self, section: str, images: dict[str, torch.Tensor]) -> None:
        try:
            self.logger.log_image(
                section, list(images.values()), caption=list(images.keys()))
        except AttributeError:
            iteration = self.global_step
            for k, v in images.items():
                cv2.imwrite(f"{section}_{iteration}_{k}.png",
                            v.permute(1, 2, 0).cpu().numpy())

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """training_step method of Model.

        Args:
            batch (dict[str, torch.Tensor]): Batch of data, contains "LQ" and "HQ" keys
            and values of shape (batch_size, 3, H, W)
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss tensor
        """
        outputs = self.model(batch["LQ"])
        metrics = self._compute_metrics(outputs, batch["HQ"])
        self._log_metrics(metrics, "train")

        return metrics["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["LQ"])
        metrics = self._compute_metrics(outputs, batch["HQ"])
        self._log_metrics(metrics, "val", sync_dist=True)

        middle_frame = batch["LQ"] if self.num_frames == 1 else batch["LQ"][:, self.num_frames // 2, :]

        bilinear = nn.functional.interpolate(
            middle_frame, scale_factor=4, mode="bilinear", align_corners=False)
        bilinear_metrics = self._compute_metrics(
            bilinear, batch["HQ"], compute_loss=False)
        self._log_metrics(bilinear_metrics, "val_upsampled", sync_dist=True)

        self._log_images(
            "val/out", {"HQ": batch["HQ"][0], "HQ Prediction": outputs[0], "Bilinear": bilinear[0]})
        return metrics["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
