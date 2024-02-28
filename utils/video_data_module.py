from typing import Any
from torch.utils.data import DataLoader
from lightning import LightningDataModule


class VideoLightningDataModule(LightningDataModule):
    def __init__(self, args: dict[str, Any]):
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        if self.args["num_frames"] > 1:
            from utils.dataset import VideoMultiFrameDataset as Dataset
        elif self.args["num_frames"] == 1:
            from utils.dataset import VideoSingleFrameDataset as Dataset

        self.train = Dataset(
            self.args["train_low_quality"], self.args["train_high_quality"], num_frames=self.args["num_frames"])
        self.val = Dataset(
            self.args["val_low_quality"], self.args["val_high_quality"], num_frames=self.args["num_frames"])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], shuffle=False)

    def test_dataloader(self):
        raise NotImplementedError
