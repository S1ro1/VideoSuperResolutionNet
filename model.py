import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch import nn, optim
from model_zoo.srresnet import srresnet_x4

torch.set_float32_matmul_precision('medium')


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = srresnet_x4()
        self.save_hyperparameters()

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
        loss = nn.functional.mse_loss(outputs, batch["HQ"])
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["LQ"])
        loss = nn.functional.mse_loss(outputs, batch["HQ"])
        self.log("val/loss", loss, prog_bar=True, logger=True)
        self.logger.log_image("val/out", [batch["HQ"][0], outputs[0]], caption=["HQ Original", "HQ Prediction"])
        return loss

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


