import json
from typing import Any, List
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
import lightning as L
import argparse

from src.video_lightning_module import VideoSRLightningModule
from src.video_data_module import VideoLightningDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, help="Path to config file")
    return parser.parse_args()


def _setup_logger(args: dict[str, Any]) -> Logger:
    if args["logger"] == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=args["project"], name=args["name"])
    elif args["logger"] == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir="logs", name=args["name"])
    else:
        raise ValueError(f"Unknown logger: {args['logger']}")

    return logger


def _setup_callbacks(args: dict[str, Any]) -> List[L.Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args['project']}_{args['name']}",
        save_top_k=args["save_top_k"],
        save_last=True,
        verbose=True,
        monitor="val/loss",
        mode="min",
    )

    return [checkpoint_callback]


def _setup_trainer(args: dict[str, Any], logger: Logger, callbacks: List[L.Callback]) -> L.Trainer:
    return L.Trainer(
        max_epochs=args["num_epochs"],
        devices=args["devices"],
        log_every_n_steps=args["log_every_n_steps"],
        logger=logger,
        check_val_every_n_epoch=args["check_val_every_n_epochs"],
        callbacks=callbacks,
        limit_train_batches=args["limit_batches"],
        limit_val_batches=args["limit_batches"],
        precision=32,
    )


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        args = json.load(f)

    logger = _setup_logger(args["misc"])
    logger.log_hyperparams(args)
    data_module = VideoLightningDataModule(args["datamodule"])
    lightning_module = VideoSRLightningModule(
        args["lightningmodule"], num_frames=args["datamodule"]["num_frames"], padding=args["transforms"]["pad"]
    )

    logger.watch(lightning_module)
    trainer = _setup_trainer(args["training"], logger, _setup_callbacks(args["misc"]))

    if args["training"].get("ckpt_path") is not None:
        trainer.fit(model=lightning_module, datamodule=data_module, ckpt_path=args["training"]["ckpt_path"])
    else:
        trainer.fit(model=lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()
