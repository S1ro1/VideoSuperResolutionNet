import wandb
import torch
from dataset import VideoSingleFrameDataset
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch import nn, optim

torch.set_float32_matmul_precision('high')

class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=4, padding=0)
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
        self.logger.log_image("val/out", [batch["LQ"][0], batch["HQ"][0], outputs[0]], caption=["LQ", "HQ Original", "HQ Prediction"])
        return loss

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    logger = WandbLogger(project="BP")

    dataset = VideoSingleFrameDataset("data/REDS/train/train_sharp_bicubic/X4", "data/REDS/train/train_sharp")
    val_dataset = VideoSingleFrameDataset("data/REDS/val/val_sharp_bicubic/X4", "data/REDS/val/val_sharp")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    model = Model()

    logger.watch(model)

    trainer = L.Trainer(limit_train_batches=100, max_epochs=1, devices=[0, 1], log_every_n_steps=1, logger=logger)
    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
