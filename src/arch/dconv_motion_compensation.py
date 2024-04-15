from arch.utils import make_conv_relu


import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d


class DConvMotionCompensation(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, num_scales: int = 3, deformable_groups=1, use_convs: bool = True
    ):
        """Deformable convolutional motion compensation block

        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            kernel_size (int, optional): Kernel size. Defaults to 3.
            num_scales (int, optional): Number of scales of preprocessing convolutions, only works if use_convs is True. Defaults to 3.
            deformable_groups (int, optional): Deformable groups. Defaults to 1.
            use_convs (bool, optional): Whether to use convolutions to preprocess features. Defaults to True.
        """
        super(DConvMotionCompensation, self).__init__()
        self.kernel_size = kernel_size
        if use_convs:
            self.convs = make_conv_relu(in_channels, in_channels, num_scales=num_scales)
        else:
            self.convs = nn.Identity()

        self.dconv = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, deformable_groups=deformable_groups
        )

    def _adjust_optical_flow(self, of: torch.Tensor) -> torch.Tensor:
        """Repeats the optical flow so it fits into the offset of deformable convolution layer with kernel size `kernel_size`

        Args:
            of (torch.Tensor): Optical flow tensor with shape (B, 2, H, W)

        Returns:
            torch.Tensor: Adjusted optical flow tensor with shape (B, kernel_size*kernel_size*2, H, W)
        """
        B, _, H, W = of.shape
        device = of.device
        of_flat = of.repeat(1, self.kernel_size * self.kernel_size, 1, 1).view(B, -1, H * W)
        conv_offsets = (
            torch.tensor(
                [
                    [(i, j) for j in range(-(self.kernel_size // 2), self.kernel_size // 2 + 1)]
                    for i in range(-(self.kernel_size // 2), self.kernel_size // 2 + 1)
                ],
                device=device,
            )
            .view(-1, 2)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        conv_offsets_flat = conv_offsets.unsqueeze(-1).expand(-1, -1, -1, H * W).view(B, -1, H * W)
        adjusted_of = of_flat - conv_offsets_flat

        return adjusted_of.view(B, self.kernel_size * self.kernel_size * 2, H, W)

    def forward(self, x: torch.Tensor, of: torch.Tensor) -> torch.Tensor:
        """Forward method

        Args:
            x (torch.Tensor): Batch of input features with shape (B, C, H, W), to be aligned by optical flow
            of (torch.Tensor): Batch of optical flow with shape (B, 2, H, W), used as offsets inside deformable convoltional layer

        Returns:
            torch.Tensor: Batch of aligned features with shape (B, C, H, W)
        """
        x = self.convs(x)
        x = x.contiguous()
        of = self._adjust_optical_flow(of)
        out = self.dconv(x, of)
        return out
