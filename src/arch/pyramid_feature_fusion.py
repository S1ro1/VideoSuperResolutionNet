# Author: Matej Sirovatka

from arch.motion_compensated_feature_fusion import MotionCompensatedFeatureFusion


import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List


class PyramidMotionCompensatedFeatureFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels_scale: int,
        out_channels: int,
        num_scales: int = 3,
        center_frame_idx: int = 1,
        use_previous: bool = False,
        use_convs: bool = True,
        use_middle_frame: bool = True,
        learnable_of: bool = False,
    ):
        """Pyramid motion compensation feature fusion block

        Args:
            in_channels (int): Input channels
            mid_channels_scale (int): Intermediate channels scale, `mid_channels = in_channels * mid_channels_scale`
            out_channels (int): Output channels
            num_scales (int, optional): Number of scales to use in feature refinement, valid only if `use_convs` is set to True. Defaults to 3.
            center_frame_idx (int, optional): Center frame index. Defaults to 1.
            use_previous (bool, optional): Unimplemented. Defaults to False.
            use_convs (bool, optional): If set to True, features are refined with convolutional layers. Defaults to True.
            use_middle_frame (bool, optional): If set to True, middle frame is used to predict the aligned features. Defaults to True.
            use_previous (bool, optional): If set to True, previous level features are used to predict the aligned features. Defaults to False.
        """
        super(PyramidMotionCompensatedFeatureFusion, self).__init__()
        self.skip_connections = nn.ModuleDict()
        self.use_previous = use_previous

        if self.use_previous:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.prev_fusions = nn.ModuleDict()
            self.prev_convs = nn.ModuleDict()

        self.num_levels = len(in_channels)

        for level, (ic, oc) in enumerate(zip(in_channels, out_channels)):
            level_name = f"level_{level}"
            self.skip_connections[level_name] = MotionCompensatedFeatureFusion(
                in_channels=ic,
                mid_channels=ic * mid_channels_scale,
                out_channels=oc,
                num_scales=num_scales,
                center_frame_idx=center_frame_idx,
                use_convs=use_convs,
                use_middle_frame=use_middle_frame,
                learnable_of=learnable_of,
            )
            if self.use_previous and level < self.num_levels - 1:
                self.prev_fusions[level_name] = nn.Conv2d(2 * oc, oc, 3, 1, 1)
                self.prev_convs[level_name] = nn.Conv2d(out_channels[level + 1], oc, 3, 1, 1)

    def forward(self, xs: List[torch.Tensor], of: torch.Tensor) -> List[torch.Tensor]:
        """Forward method

        Args:
            xs (List[torch.Tensor]): List of input feature maps in shape (B, T, C, H, W)
            of [torch.Tensor]: Tensor of optical flow in shape (B, T, 2, H, W)

        Returns:
            List[torch.Tensor]: List of aligned feature maps in shape (B, C, H, W)
        """
        self.ofs = []
        b, t, c, h, w = of.shape
        for i, (_, _, _, h_, w_) in enumerate([x.shape for x in xs]):
            of_ = F.interpolate(of.view(b * t, c, h, w), size=(h_, w_), mode="bilinear", align_corners=False)
            self.ofs.append(of_.view(b, t, c, h_, w_) / (2.0**i))

        out = []

        prev_features = None

        for i, (x, of) in enumerate(zip(xs[::-1], self.ofs[::-1])):
            level = self.num_levels - i - 1
            level_name = f"level_{level}"
            features = self.skip_connections[level_name](x, of)

            if self.use_previous and prev_features is not None:
                prev = self.prev_convs[level_name](self.upsample(prev_features))
                features = self.prev_fusions[level_name](torch.cat([features, prev], dim=1))

            prev_features = features

            out.append(features)

        return out[::-1]
