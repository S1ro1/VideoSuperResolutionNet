import torch.nn as nn
from basicsr.archs.arch_util import DCNv2Pack
from typing import List
import torch


class MotionCompensationBlock(nn.Module):
    def __init__(self, num_features: int, scale_factor: int = 2, num_layers: int = 3):
        super(MotionCompensationBlock, self).__init__()
        self.layer_channels = [num_features * scale_factor**i for i in range(num_layers)]
        self.offset_convs = nn.ModuleList()
        self.dconvs = nn.ModuleList()
        self.feat_convs = nn.ModuleList()
        self.channel_convs = nn.ModuleList()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

        for i in range(num_layers):
            # num_channels = num_features * scale_factor ** i
            self.offset_convs.append(nn.Conv2d(self.layer_channels[i] * 2 + 2, self.layer_channels[i], 3, 1, 1))
            self.dconvs.append(DCNv2Pack(self.layer_channels[i], self.layer_channels[i], 3, padding=1, deformable_groups=8))
            if i < num_layers - 1:
                self.feat_convs.append(nn.Conv2d(self.layer_channels[i] * 2, self.layer_channels[i], 3, 1, 1))
                self.channel_convs.append(nn.Conv2d(self.layer_channels[i + 1], self.layer_channels[i], 3, 1, 1))

    def forward(self, x: List[torch.Tensor], optical_flows: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass

        Args:
            x (List[torch.Tensor]): List of input features where each tensor has shape (B, T, C, H, W), where T == 1 is neighboring frame, and H, W
            are the spatial dims (each element in the list is a different pyramid level, spatial dims decrease with pyramid level) and C is the number of features (grows with pyramid level)
            optical_flows (List[torch.Tensor]): list of optical flows where each tensor has shape (B, 2, H, W) and H, W are the spatial dims (each element in the list is a different pyramid level, spatial dims decrease with pyramid level)

        Returns:
            torch.Tensor: output feature map
        """
        aligned_features = []
        prev_features = None

        for level in range(len(x) - 1, -1, -1):
            B, T, C, H, W = x[level].size()
            offset = torch.cat([x[level].view(B, T * C, H, W), optical_flows[level]], dim=1)
            offset = self.offset_convs[level](offset)

            neighboring_frame = x[level][:, 1, ...]
            features = self.dconvs[level](neighboring_frame, offset)
            if level < len(x) - 1:
                low_channel_prev_features = self.channel_convs[level](prev_features)
                features = torch.cat([features, low_channel_prev_features], dim=1)
                features = self.feat_convs[level](features)

            prev_features = self.upsample(features)
            aligned_features.append(features)

        return aligned_features[::-1]


class MotionCompensatedFeatureFusion(nn.Module):
    def __init__(self, num_features: int, num_frames: int = 3, num_layers: int = 3):
        super(MotionCompensatedFeatureFusion, self).__init__()
        self.num_frames = num_frames
        self.num_features = num_features
        self.motion_compensation_block = MotionCompensationBlock(num_features, num_layers=num_layers)

        if self.num_frames != 3:
            raise NotImplementedError("Only 3 frames are supported")

        self.fusion_convs = nn.ModuleList()

        for i in range(num_layers):
            n_features = num_features * 2**i
            self.fusion_convs.append(nn.Conv2d(n_features * 2, n_features, 3, 1, 1))

    def forward(self, x: List[torch.Tensor], optical_flows: List[torch.Tensor]) -> torch.Tensor:
        # x = list of input features [B, T, C, H, W]
        # of = list of optical flows [B, T - 1, 2, H, W]
        forward_flows = [optical_flow[:, 0, ...] for optical_flow in optical_flows]
        backward_flows = [optical_flow[:, 1, ...] for optical_flow in optical_flows]

        frame1, frame2, frame3 = [f[:, 0, ...] for f in x], [f[:, 1, ...] for f in x], [f[:, 2, ...] for f in x]

        forward_aligned_features = self.motion_compensation_block(
            [torch.stack([frame3[i], frame2[i]], dim=1) for i in range(len(x))], forward_flows
        )
        backward_aligned_features = self.motion_compensation_block(
            [torch.stack([frame1[i], frame2[i]], dim=1) for i in range(len(x))], backward_flows
        )
        features = [torch.cat([forward_aligned_features[i], backward_aligned_features[i]], dim=1) for i in range(len(x))]
        features = [self.fusion_convs[i](features[i]) for i in range(len(x))]

        return features
