from typing import Literal
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
        return {"LQ": torch.stack(lq_frames), "HQ": torch.stack(hq_frames)}


class VideoSingleFrameDataset(Dataset):
    def __init__(self, lq_path: str, hq_path: str, *_args, **_kwargs):
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
        return {"LQ": torchvision.io.read_image(self.lq_frames[idx]) / 255.0, "HQ": torchvision.io.read_image(self.hq_frames[idx]) / 255.0}


class VideoMultiFrameDataset(Dataset):
    def __init__(self, lq_path: str, hq_path: str, num_frames: int, *_args, **_kwargs):
        """Dataset for multi frame training,
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
            num_frames (int): number of frames to be stacked in 1 element
        """
        self.num_frames = num_frames

        lq_sequences = sorted(list(Path(lq_path).iterdir()))
        hq_sequences = sorted(list(Path(hq_path).iterdir()))

        assert len(lq_sequences) == len(hq_sequences), "Number of low quality and high quality sequences must be equal."

        self.lq_paths = []
        self.hq_paths = []

        for lq_sequence_path, hq_sequence_path in zip(lq_sequences, hq_sequences):
            lq_frame_paths = sorted(list(lq_sequence_path.iterdir()))
            hq_frame_paths = sorted(list(hq_sequence_path.iterdir()))

            lq_frame_sequences = [lq_frame_paths[i : i + self.num_frames] for i in range(len(lq_frame_paths) - self.num_frames + 1)]

            hq_frame_sequences = [hq_frame_paths[i + self.num_frames // 2] for i in range(len(hq_frame_paths) - self.num_frames + 1)]

            self.lq_paths.extend(lq_frame_sequences)
            self.hq_paths.extend(hq_frame_sequences)

        assert len(self.lq_paths) == len(self.hq_paths), "Number of low quality and high quality frames must be equal."

    def __len__(self) -> int:
        """Number of sequences with length self.num_frames in the dataset

        Returns:
            int: Number of sequences with length self.num_frames in the dataset
        """
        return len(self.lq_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """__getitem__ method of VideoMultiFrameDataset.

        Args:
            idx (int): Index of the sequence to be returned.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys "LQ" and "HQ" and values
            being tensors of shape (T, C, H, W) for "LQ" and (C, H, W) for "HQ" where T is self.num_frames
        """
        lq_frame_paths = self.lq_paths[idx]
        hq_frame_path = self.hq_paths[idx]

        lq_frames = []

        for lq_frame_path in lq_frame_paths:
            lq_frames.append(torchvision.io.read_image(str(lq_frame_path)) / 255.0)

        lq_frames = torch.stack(lq_frames)
        hq_frame = torchvision.io.read_image(str(hq_frame_path)) / 255.0
        if self.transforms is not None:
            lq_frames = self.transforms(lq_frames)

        return {"LQ": lq_frames, "HQ": hq_frame}


class VideoMultiFrameOFDataset(Dataset):
    def __init__(
        self,
        lq_path: str,
        hq_path: str,
        of_path: str,
        num_frames: int = 3,
        of_type: Literal["calculated", "zero", "random", "minus"] = "calculated",
        *_args,
        **_kwargs,
    ):
        """Dataset for multi frame training with optical flow,
        expects directory structure to be:
        lq_path:
        - sequence_1
        -- frame_1
        -- frame_n
        - sequence_n

        hq_path:
        ...

        of_path:
        - sequence_1
        -- frame_1_down.pt
        -- frame_1_up.pt
        -- frame_n_down.pt

        Args:
            lq_path (str): path to low quality frames
            hq_path (str): path to high quality frames
            of_path (str): path to optical flow frames
            num_frames (int): number of frames to be stacked in 1 element
            of_type (Literal["calculated", "zero", "random"]): Type of optical flow to be used (default: "calculated")

        Raises:
            NotImplementedError: If num_frames is not 3
        """
        self.num_frames = num_frames
        self.of_type = of_type

        if self.num_frames != 3:
            raise NotImplementedError("Only 3 frames are supported for now.")

        lq_sequences = sorted(list(Path(lq_path).iterdir()))
        hq_sequences = sorted(list(Path(hq_path).iterdir())) if hq_path is not None else [None] * len(lq_sequences)
        of_sequences = sorted(list(Path(of_path).iterdir()))

        assert len(lq_sequences) == len(hq_sequences), "Number of low quality and high quality sequences must be equal."

        self.lq_paths = []
        self.hq_paths = []
        self.of_up_paths = []
        self.of_down_paths = []

        for lq_sequence_path, hq_sequence_path, of_sequence_path in zip(lq_sequences, hq_sequences, of_sequences):
            lq_frame_paths = sorted(list(lq_sequence_path.iterdir()))
            hq_frame_paths = sorted(list(hq_sequence_path.iterdir())) if hq_sequence_path is not None else [None] * len(lq_frame_paths)

            lq_frame_sequences = [lq_frame_paths[i : i + self.num_frames] for i in range(len(lq_frame_paths) - self.num_frames + 1)]
            hq_frame_sequences = (
                [hq_frame_paths[i + self.num_frames // 2] for i in range(len(hq_frame_paths) - self.num_frames + 1)]
                if hq_sequence_path is not None
                else [None] * len(lq_frame_sequences)
            )

            self.lq_paths.extend(lq_frame_sequences)
            self.hq_paths.extend(hq_frame_sequences)

            middle_frames = [lq_frame_paths[i + self.num_frames // 2].stem for i in range(len(lq_frame_paths) - self.num_frames + 1)]
            of_frames_down = [of_sequence_path / f"{frame}_down.pt" for frame in middle_frames]
            of_frames_up = [of_sequence_path / f"{frame}_up.pt" for frame in middle_frames]

            self.of_down_paths.extend(of_frames_down)
            self.of_up_paths.extend(of_frames_up)

        assert len(self.lq_paths) == len(self.hq_paths), "Number of low quality and high quality frames must be equal."

    def __len__(self) -> int:
        """__len__ method of VideoMultiFrameOFDataset.

        Returns:
            int: Number of sequences with length self.num_frames in the dataset
        """
        return len(self.lq_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """__getitem__ method of VideoMultiFrameOFDataset.

        Args:
            idx (int): Index of the sequence to be returned.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys "LQ", "HQ", "OF_down" and "OF_up" and values
            being tensors of shape (T, C, H, W) for "LQ" and (C, H, W) for "HQ" where T is self.num_frames
            and (2, H, W) for "OF_down" and "OF_up", OF_up and OF_down are optical flows between the middle LQ frame and next, previous HQ frame respectively.
        """
        lq_frame_paths = self.lq_paths[idx]
        hq_frame_path = self.hq_paths[idx]
        of_down_path = self.of_down_paths[idx]
        of_up_path = self.of_up_paths[idx]

        lq_frames = []

        for lq_frame_path in lq_frame_paths:
            lq_frames.append(torchvision.io.read_image(str(lq_frame_path)) / 255.0)

        if self.of_type == "calculated" or self.of_type == "minus":
            of_down = torch.load(of_down_path, map_location="cpu").squeeze(0)
            of_up = torch.load(of_up_path, map_location="cpu").squeeze(0)
        elif self.of_type == "zero":
            of_down = torch.zeros(2, lq_frames[0].shape[-2], lq_frames[0].shape[-1])
            of_up = torch.zeros(2, lq_frames[0].shape[-2], lq_frames[0].shape[-1])
        elif self.of_type == "random":
            of_down = torch.rand(2, lq_frames[0].shape[-2], lq_frames[0].shape[-1])
            of_up = torch.rand(2, lq_frames[0].shape[-2], lq_frames[0].shape[-1])

        if self.of_type == "minus":
            of_down = -of_down
            of_up = -of_up

        # Crop optical flow to match the size of the frames (RAFT pads the image symmetrically in sintel mode)
        hd = of_up.shape[-2] - lq_frames[0].shape[-2]
        wd = of_up.shape[-1] - lq_frames[0].shape[-1]

        if hd != 0:
            of_down = of_down[..., hd // 2 : -hd // 2, :]
            of_up = of_up[..., hd // 2 : -hd // 2, :]
        if wd != 0:
            of_down = of_down[..., wd // 2 : -wd // 2]
            of_up = of_up[..., wd // 2 : -wd // 2]

        lq_frames = torch.stack(lq_frames)
        hq_frame = torchvision.io.read_image(str(hq_frame_path)) / 255.0 if self.hq_paths[0] is not None else torch.zeros_like(lq_frames[0])
        ofs = torch.stack([of_down, of_up])

        return {"LQ": lq_frames, "HQ": hq_frame, "OF": ofs}
