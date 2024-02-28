
import json
from typing import Any
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
import lightning as L
import argparse

from utils.video_lightning_module import VideoSRLightningModule
from utils.video_data_module import VideoLightningDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, help="Path to config file")
    return parser.parse_args()


def _setup_logger(args: dict[str, Any]) -> Logger:
    if args["logger"] == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(
            project=args["project"], name=args["name"])
    elif args["logger"] == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir="logs", name=args["name"])

    return logger


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        args = json.load(f)

    logger = _setup_logger(args["misc"])

    data_module = VideoLightningDataModule(args["datamodule"])
    model = VideoSRLightningModule(
        args["lightningmodule"], num_frames=args["datamodule"]["num_frames"])

    logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args['misc']['project']}_{args['misc']['name']}",
        save_top_k=args["misc"]["save_top_k"], save_last=True, verbose=True,
        monitor="val/loss", mode="min")

    training_args = args["training"]
    trainer = L.Trainer(
        max_epochs=training_args["num_epochs"],
        devices=training_args["devices"],
        log_every_n_steps=training_args["log_every_n_steps"],
        logger=logger,
        check_val_every_n_epoch=training_args["check_val_every_n_epochs"],
        callbacks=[checkpoint_callback],
        limit_train_batches=training_args["limit_batches"],
        limit_val_batches=training_args["limit_batches"],
        precision=32,
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
