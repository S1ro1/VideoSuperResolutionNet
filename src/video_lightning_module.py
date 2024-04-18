from typing import Any, List, Optional
import torch
import lightning as L
from torch import nn, optim
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import cv2

torch.set_float32_matmul_precision("medium")


class VideoSRLightningModule(L.LightningModule):
    def __init__(self, args: dict[str, Any], num_frames: int = 1, padding: Optional[List[int]] = None):
        """LightningModule for video super-resolution

        Args:
            args (dict[str, Any]): Arguments for the model
            num_frames (int, optional): Number of frames to use. Defaults to 1.
            padding (Optional[List[int]], optional): List of paddings that are required by the model. Defaults to None.
        """
        super().__init__()
        self.num_frames = num_frames
        self.args = args
        self.padding = padding
        self._setup()
        self._setup_input_padder()
        self.save_hyperparameters()

    def _setup(self) -> None:
        """Factory method to setup the model

        Raises:
            NotImplementedError: If the model is not implemented
        """
        self.lr = self.args["lr"]
        if self.args["model_name"] == "srresnet":
            from arch.srresnet import srresnet_x4 as Model
        elif self.args["model_name"] == "EDVR":
            from basicsr.archs.edvr_arch import EDVR as Model
        elif self.args["model_name"] == "UNet":
            from arch.unet import SuperResolutionUnet as Model
        else:
            raise NotImplementedError(f"Model {self.args['model_name']} has not been implemented yet.")

        self.model = Model(**self.args["model_args"])

    def _setup_input_padder(self):
        """Setup input padder and cropper for the model"""
        if self.padding is not None:
            self.input_padder = lambda x: nn.functional.pad(x, self.padding)
            self.input_crop = lambda x: x[
                :,
                :,
                self.padding[2] * 4 : -self.padding[3] * 4 if self.padding[3] else None,
                self.padding[0] * 4 : -self.padding[1] * 4 if self.padding[1] else None,
            ]

    def _compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, compute_loss: bool = True) -> dict[str, torch.Tensor]:
        """Compute metrics for the model (ssim, psnr, l2 loss)

        Args:
            outputs (torch.Tensor): Outputs of the model
            targets (torch.Tensor): Target predictions
            compute_loss (bool, optional): Whether loss should be computed for optimization. Defaults to True.

        Returns:
            dict[str, torch.Tensor]: Dictionary of metrics with keys "ssim", "psnr" and "loss" optionally
        """
        metrics = {}
        metrics["ssim"] = structural_similarity_index_measure(outputs, targets)
        metrics["psnr"] = peak_signal_noise_ratio(outputs, targets)
        if compute_loss:
            metrics["loss"] = nn.functional.mse_loss(outputs, targets)
        return metrics

    def _log_metrics(self, metrics: dict[str, torch.Tensor], prefix: str, sync_dist: bool = False) -> None:
        """Log metrics to the logger

        Args:
            metrics (dict[str, torch.Tensor]): Dictionary of metrics
            prefix (str): Prefix for the metrics, such as "train" or "val"
            sync_dist (bool, optional): For multi-gpu. Defaults to False.
        """
        for key, value in metrics.items():
            self.log(f"{prefix}/{key}", value, prog_bar=True, logger=True, sync_dist=sync_dist)

    def _log_images(self, section: str, images: dict[str, torch.Tensor]) -> None:
        try:
            self.logger.log_image(section, list(images.values()), caption=list(images.keys()))
        except AttributeError:
            iteration = self.global_step
            for k, v in images.items():
                cv2.imwrite(f"{section}_{iteration}_{k}.png", v.permute(1, 2, 0).cpu().numpy())

    def _common_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Common step for training and validation

        Args:
            batch (dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Output of the model
        """
        x = self.input_padder(batch["LQ"]) if self.padding is not None else batch["LQ"]

        if self.args.get("use_optical_flow", True):
            of = self.input_padder(batch["OF"]) if self.padding is not None else batch["OF"]
            x = {"LQ": x, "OF": of}

        outputs = self.model(x)

        if self.padding is not None:
            outputs = self.input_crop(outputs)

        return outputs

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """training_step method of Model.

        Args:
            batch (dict[str, torch.Tensor]): Batch of data, contains "LQ" and "HQ" keys, optionally "OF" key
            and values of shape (batch_size, 3, H, W) or (batch_size, T, 3, H, W) respectively.
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss tensor
        """
        outputs = self._common_step(batch, batch_idx)

        metrics = self._compute_metrics(outputs, batch["HQ"])
        self._log_metrics(metrics, "train")

        return metrics["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """validation_step method of Model.

        Args:
            batch (dict[str, torch.Tensor]): Batch of data, contains "LQ" and "HQ" keys, optionally "OF" key
            and values of shape (batch_size, 3, H, W) or (batch_size, T, 3, H, W) respectively.
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss tensor
        """
        outputs = self._common_step(batch, batch_idx)

        metrics = self._compute_metrics(outputs, batch["HQ"])
        self._log_metrics(metrics, "val", sync_dist=True)

        middle_frame = batch["LQ"] if self.num_frames == 1 else batch["LQ"][:, self.num_frames // 2, :]
        bilinear = nn.functional.interpolate(middle_frame, scale_factor=4, mode="bilinear", align_corners=False)
        bilinear_metrics = self._compute_metrics(bilinear, batch["HQ"], compute_loss=False)
        self._log_metrics(bilinear_metrics, "val_upsampled", sync_dist=True)
        if batch_idx == len(self.trainer.val_dataloaders) - 1:
            for img_idx in range(len(batch["HQ"])):
                self._log_images("val/out", {"HQ": batch["HQ"][img_idx], "HQ Prediction": outputs[img_idx], "Bilinear": bilinear[img_idx]})

        return metrics["loss"]
    
    def predict_step(self, batch, batch_idx) -> Any:
        out = self._common_step(batch, batch_idx)
        return out

    def configure_optimizers(self) -> None:
        """Configure optimizers for the model"""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return {"optimizer": optimizer, "lr_scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)}
