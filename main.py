
from dataset import VideoSingleFrameDataset
import torch
import torch.utils.data
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L

from model import Model
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--project", type=str, help="Logger project name.")
    parser.add_argument("--name", type=str, help="Logger experiment name.")
    parser.add_argument("--logger-type", type=str, default="wandb",
                        help="Logger type (wandb, tensorboard)")

    parser.add_argument("--train-hq", type=str, required=True,
                        help="Path to the high quality training dataset")
    parser.add_argument("--train-lq", type=str, required=True,
                        help="Path to the low quality training dataset")
    parser.add_argument("--val-hq", type=str, required=True,
                        help="Path to the high quality validation dataset")
    parser.add_argument("--val-lq", type=str, required=True,
                        help="Path to the low quality validation dataset")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=6,
                        help="Number of workers for the dataloader")

    parser.add_argument("--devices", type=int, nargs="+",
                        default=[0], help="Devices to use for training")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train")
    parser.add_argument("--log-every-n-steps", type=int,
                        default=50, help="Log every n steps")
    parser.add_argument("--check-val-every-n-epoch", type=int,
                        default=5, help="Check validation every n epochs")
    parser.add_argument("--save-top-k", type=int,
                        default=3, help="Save top k models")

    parser.add_argument("--limit-batches", type=float, default=1.0,
                        help="Limit the number of batches to use for training and validation")

    parser.add_argument("--learnging-rate", type=float,
                        default=2e-3, help="Learning rate")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.logger_type == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project=args.project, name=args.name)
    elif args.logger_type == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir="logs", name=args.name)

    train_dataset = VideoSingleFrameDataset(args.train_lq, args.train_hq)
    val_dataset = VideoSingleFrameDataset(args.val_lq, args.val_hq)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    model = Model(lr=args.learnging_rate)
    # logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.project}_{args.name}",
        save_top_k=args.save_top_k, save_last=True, verbose=True,
        monitor="val/loss", mode="min")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
    )

    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
