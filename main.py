from lightning.pytorch.loggers import WandbLogger
from dataset import VideoSingleFrameDataset
import torch
import torch.utils.data
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L

from model import Model

if __name__ == "__main__":
    logger = WandbLogger(project="BP")
    train_dataset = VideoSingleFrameDataset("data/REDS/train/train_sharp_bicubic/X4", "data/REDS/train/train_sharp")
    val_dataset = VideoSingleFrameDataset("data/REDS/val/val_sharp_bicubic/X4", "data/REDS/val/val_sharp")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=31, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=32, shuffle=True)

    model = Model()
    logger.watch(model)

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/srresnet", save_top_k=3, monitor="val/loss")
    trainer = L.Trainer(
        max_epochs=100,
        devices=[0, 1],
        log_every_n_steps=50,
        logger=logger,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
