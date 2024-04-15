from arch.dconv_motion_compensation import DConvMotionCompensation
from arch.utils import make_conv_relu


import torch
import torch.nn as nn


class MotionCompensatedFeatureFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_convs: bool = True,
        num_scales: int = 3,
        center_frame_idx: int = 1,
        use_middle_frame: bool = True,
    ):
        """Motion compensation feature fusion block

        Args:
            in_channels (int): Input channels
            mid_channels (int): Intermediate channels used in feature refinement
            out_channels (int): Output channels
            use_convs (bool, optional): If set to True, features are further refined using convolutional layers. Defaults to True.
            num_scales (int, optional): If `use_convs` is set to True, number of scales in the feature refinement. Defaults to 3.
            center_frame_idx (int, optional): Center frame index. Defaults to 1.
            use_middle_frame (bool, optional): If set to True, middle frame is used to predict aligned features. Defaults to True.
        """
        super(MotionCompensatedFeatureFusion, self).__init__()
        if use_convs and use_middle_frame:
            ref_convs = make_conv_relu(in_channels, in_channels, num_scales=num_scales)
            end_ref_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
            self.ref_convs = nn.Sequential(*ref_convs, end_ref_conv)
        elif not use_convs and use_middle_frame:
            assert in_channels == mid_channels, "If not using convs, in_channels must be equal to mid_channels."
            self.ref_convs = nn.Identity()
        elif not use_convs and not use_middle_frame:
            self.ref_convs = None
        elif use_convs and not use_middle_frame:
            self.ref_convs = None

        self.neighboring_blocks = DConvMotionCompensation(
            in_channels=in_channels, out_channels=mid_channels, num_scales=num_scales, use_convs=use_convs
        )
        self.center_frame_idx = center_frame_idx
        self.use_middle_frame = use_middle_frame
        feature_count = 3 if use_middle_frame else 2

        self.fusion_block = nn.Sequential(
            nn.Conv2d(mid_channels * feature_count, mid_channels * feature_count, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * feature_count, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, of: torch.Tensor) -> torch.Tensor:
        """Forward method

        Args:
            x (torch.Tensor): Torch tensor of input features with shape (B, T, C, H, W)
            of (torch.Tensor): Torch tensor of optical flow with shape (B, T, 2, H, W)

        Returns:
            torch.Tensor: Torch tensor of aligned features with shape (B, C, H, W)
        """
        if self.use_middle_frame:
            ref = x[:, self.center_frame_idx, ...]
            ref_out = self.ref_convs(ref)
            neighbor_indices = [i for i in range(x.size(1)) if i != self.center_frame_idx]
        else:
            neighbor_indices = [i for i in range(x.size(1))]

        of_indices = [i for i in range(of.size(1))]
        neighbor_out = []
        for (
            ni,
            oi,
        ) in zip(neighbor_indices, of_indices):
            neighbor_out.append(self.neighboring_blocks(x[:, ni, ...], of[:, oi, ...]))

        neighbor_out = torch.cat(neighbor_out, dim=1)

        out = torch.cat([ref_out, neighbor_out], dim=1) if self.use_middle_frame else neighbor_out

        return self.fusion_block(out)
