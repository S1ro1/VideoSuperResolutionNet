import torch
import torchvision
from pathlib import Path
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, lq_dir: str, hq_dir: str):
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
            lq_frames.append(torchvision.io.read_image(str(lq_frame_path)))
            hq_frames.append(torchvision.io.read_image(str(hq_frame_path)))
        
        assert len(lq_frames) == len(hq_frames), "Number of low quality and high quality frames must be equal."
        return {
            "LQ": torch.stack(lq_frames),
            "HQ": torch.stack(hq_frames)
        }
