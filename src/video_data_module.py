# Author: Matej Sirovatka

from typing import Any
from torch.utils.data import DataLoader
from lightning import LightningDataModule


class VideoLightningDataModule(LightningDataModule):
    def __init__(self, args: dict[str, Any]):
        """LightningDataModule for video data

        Args:
            args (dict[str, Any]): Arguments for the data module
        """
        super().__init__()
        self.args = args

    def setup(self, stage: str):
        """Setup the data module, serves as factory method

        Args:
            stage (str): Stage of the data module
        """
        if self.args["use_optical_flow"]:
            from dataset import VideoMultiFrameOFDataset as Dataset
        elif self.args["num_frames"] > 1:
            from dataset import VideoMultiFrameDataset as Dataset
        elif self.args["num_frames"] == 1:
            from dataset import VideoSingleFrameDataset as Dataset

        self.train = Dataset(**self.args["train_dataset_args"])
        self.val = Dataset(**self.args["val_dataset_args"])

    def train_dataloader(self) -> DataLoader:
        """Train dataloader

        Returns:
            DataLoader: Dataloader for training
        """
        return DataLoader(self.train, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader

        Returns:
            DataLoader: Dataloader for validation
        """
        return DataLoader(self.val, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"], shuffle=False)

    def test_dataloader(self):
        """Test dataloader

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
