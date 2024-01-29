import torch
import torchvision
from pathlib import Path
from torch.utils.data import Dataset


class VideoSequenceDataset(Dataset):
    def __init__(self, lq_dir: str, hq_dir: str):
        """Dataset for training on sequences of frames,
        expects directory structure to be:
        lq_path:
        - sequence_1
        -- frame_1
        -- frame_n
        - sequence_n

        hq_path:
        ...

        Args:
            lq_path (str): path to low quality frames
            hq_path (str): path to high quality frames
        """
        self.lq_sequences = list(Path(lq_dir).iterdir())
        self.hq_sequences = list(Path(hq_dir).iterdir())
        assert len(self.lq_sequences) == len(self.hq_sequences), "Number of low quality and high quality sequences must be equal."
    
    def __len__(self) -> int:
        """Number of sequences in the dataset

        Returns:
            int: Number of sequences in the dataset
        """
        return len(self.lq_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """__getitem__ method of VideoDataset.

        Args:
            idx (int): Index of the sequence to be returned.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys "LQ" and "HQ" and values
            being tensors of shape (T, C, H, W) where T is the number of frames in the sequence
        """

        lq_frames = []
        hq_frames = []

        lq_sequence_path = self.lq_sequences[idx]
        hq_sequence_path = self.hq_sequences[idx]
        for lq_frame_path, hq_frame_path in zip(lq_sequence_path.iterdir(), hq_sequence_path.iterdir()):
            lq_frames.append(torchvision.io.read_image(str(lq_frame_path)) / 255.0)
            hq_frames.append(torchvision.io.read_image(str(hq_frame_path)) / 255.0)
        
        assert len(lq_frames) == len(hq_frames), "Number of low quality and high quality frames must be equal."
        return {
            "LQ": torch.stack(lq_frames),
            "HQ": torch.stack(hq_frames)
        }


class VideoSingleFrameDataset(Dataset):
    def __init__(self, lq_path: str, hq_path: str):
        """Dataset for single frame training,
        expects directory structure to be:
        lq_path:
        - sequence_1
        -- frame_1
        -- frame_n
        - sequence_n

        hq_path:
        ...

        Args:
            lq_path (str): path to low quality frames
            hq_path (str): path to high quality frames
        """
        lq_sequences = list(Path(lq_path).iterdir())
        hq_sequences = list(Path(hq_path).iterdir())

        assert len(lq_sequences) == len(hq_sequences), "Number of low quality and high quality sequences must be equal."

        self.lq_frames = []
        self.hq_frames = []

        for lq_sequence_path, hq_sequence_path in zip(lq_sequences, hq_sequences):
            for lq_frame_path, hq_frame_path in zip(lq_sequence_path.iterdir(), hq_sequence_path.iterdir()):
                self.lq_frames.append(str(lq_frame_path))
                self.hq_frames.append(str(hq_frame_path))
        
        assert len(self.lq_frames) == len(self.hq_frames), "Number of low quality and high quality frames must be equal."
    
    def __len__(self) -> int:
        """Number of frames in the dataset

        Returns:
            int: Number of frames in the dataset
        """
        return len(self.lq_frames)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """__getitem__ method of VideoSingleFrameDataset.

        Args:
            idx (int): Index of the frame to be returned.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys "LQ" and "HQ" and values
            being tensors of shape (C, H, W)
        """
        return {
            "LQ": torchvision.io.read_image(self.lq_frames[idx]) / 255.0,
            "HQ": torchvision.io.read_image(self.hq_frames[idx]) / 255.0
        }


if __name__ == "__main__":
    dataset = VideoSingleFrameDataset("data/REDS/train/train_sharp_bicubic/X4", "data/REDS/train/train_sharp")

    print(len(dataset))


