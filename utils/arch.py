from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock, DecoderBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from mmcv.ops import DeformConv2d
from typing import Callable, List, Optional
import inspect


def make_conv_relu(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, num_scales: int = 3
) -> nn.Sequential:
    """Creates a sequence of convolutional layers followed by ReLU activation

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride. Defaults to 1.
        padding (int, optional): Padding. Defaults to 1.
        num_scales (int, optional): Number of blocks. Defaults to 3.

    Returns:
        nn.Sequential: Sequence of convolutional layers followed by ReLU activation
    """
    convs = []
    for _ in range(num_scales):
        convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        convs.append(nn.ReLU(inplace=True))
    return nn.Sequential(*convs)


def make_layer(layer: Callable, num_layers: int, *args, **kwargs) -> nn.Sequential:
    """Creates a sequence of layers

    Args:
        layer (Callable): Layer function
        num_layers (int): Number of layers

    Returns:
        nn.Sequential: Sequence of layers
    """
    layers = []
    for _ in range(num_layers):
        layers.append(layer(*args, **kwargs))
    return nn.Sequential(*layers)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, num_features):
        """Bottleneck residual block, ---1x1 conv---3x3 conv---1x1 conv---+--->
                                       |__________________________________|
        Args:
            num_features (_type_): Number of input features
        """
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1)

    def forward(self, x):
        i = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return F.relu(out + i)


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


class MotionCompensationFeatureFusion(nn.Module):
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
        super(MotionCompensationFeatureFusion, self).__init__()
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


class PyramidMotionCompensationFetaureFusion(nn.Module):
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
        super(PyramidMotionCompensationFetaureFusion, self).__init__()
        self.skip_connections = nn.ModuleDict()
        self.use_previous = use_previous

        if self.use_previous:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.prev_fusions = nn.ModuleDict()
            self.prev_convs = nn.ModuleDict()

        self.num_levels = len(in_channels)

        for level, (ic, oc) in enumerate(zip(in_channels, out_channels)):
            level_name = f"level_{level}"
            self.skip_connections[level_name] = MotionCompensationFeatureFusion(
                in_channels=ic,
                mid_channels=ic * mid_channels_scale,
                out_channels=oc,
                num_scales=num_scales,
                center_frame_idx=center_frame_idx,
                use_convs=use_convs,
                use_middle_frame=use_middle_frame,
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


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int):
        """Upsample block, upsamples the output 4x

        Args:
            in_channels (int): Input channels
        """
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upconv1 = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.conv_hr = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(in_channels, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.pixel_shuffle(self.upconv1(x)))
        out = F.relu(self.pixel_shuffle(self.upconv2(out)))
        out = F.relu(self.conv_hr(out))
        return self.conv_last(out)


class SuperResolutionUnet(smp.Unet):
    def __init__(
        self,
        mid_conv_channels_scale: int = 2,
        num_reconstruction_blocks: Optional[int] = None,
        use_convs: bool = True,
        num_scales: int = 3,
        use_skip_connections: bool = True,
        use_middle_frame: bool = True,
        use_previous: bool = False,
        *args,
        **kwargs,
    ):
        """Super resolution U-Net model

        Args:
            mid_conv_channels_scale (int, optional): Number to scale intermediate channels with:
            `mid_channels = in_channels * mid_conv_channels_scale`. Defaults to 2
            num_reconstruction_blocks (Optional[int], optional): Number of reconstruction blocks to be used
            after contracting path. Defaults to None.
            use_convs (bool, optional): If set to True, features are refined with convolutional layers in skip connections. Defaults to True.
            num_scales (int, optional): Num scales of the refinement convolutional layers. Defaults to 3.
            use_skip_connections (bool, optional): If set to True, feature alignment skip connections are used. Defaults to True.
            use_middle_frame (bool, optional): If set to True, middle frame is used to predict the aligned features. Defaults to True.
            use_previous (bool, optional): If set to True, previous level features are used to predict the aligned features in the skip connection. Defaults to False.
        """
        super(SuperResolutionUnet, self).__init__(*args, **kwargs)
        self.use_skip_connections = use_skip_connections
        self.use_middle_frame = use_middle_frame

        sig = inspect.signature(smp.Unet.__init__)
        default_values = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

        decoder_channels = kwargs.get("decoder_channels", default_values["decoder_channels"])
        n_blocks = kwargs.get("encoder_depth", default_values["encoder_depth"])
        use_batchnorm = kwargs.get("decoder_use_batchnorm", default_values["decoder_use_batchnorm"])
        encoder_name = kwargs.get("encoder_name", default_values["encoder_name"])
        attention_type = kwargs.get("decoder_attention_type", default_values["decoder_attention_type"])

        out_channels = decoder_channels[-1]

        if use_skip_connections:
            self.middle = PyramidMotionCompensationFetaureFusion(
                in_channels=self.encoder.out_channels,
                mid_channels_scale=mid_conv_channels_scale,
                out_channels=self.encoder.out_channels,
                num_scales=num_scales,
                center_frame_idx=1,
                use_convs=use_convs,
                use_middle_frame=use_middle_frame,
                use_previous=use_previous,
            )

        self.segmentation_head = None
        self.decoder = UnetDecoderWithFirstSkip(
            self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
            use_batchnorm=use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=attention_type,
        )
        if num_reconstruction_blocks is not None:
            self.reconstruction_blocks = make_layer(BottleneckResidualBlock, num_reconstruction_blocks, out_channels)
        else:
            self.reconstruction_blocks = nn.Identity()

        self.upsample = UpsampleBlock(out_channels)

    def _multi_frame_of_forward(self, x: dict) -> torch.Tensor:
        """Forward method for multi-frame optical flow

        Args:
            x (dict): Dictionary containing keys "LQ" and "OF"
            with values being torch tensors of low quality frames and optical flow of shape (B, T, C, H, W), (B, T - 1, 2, H, W) respectively

        Returns:
            torch.Tensor: Output tensor of shape (B, 3, H, W)
        """
        feat = x["LQ"]
        of = x["OF"]

        if not self.use_middle_frame:
            feat = torch.cat([feat[:, :1], feat[:, 2:]], dim=1)

        b, t, c, h, w = feat.shape
        flat_xs = feat.view(b * t, c, h, w)
        self.check_input_shape(flat_xs)

        features = self.encoder(flat_xs)
        new_features = []
        for f in features:
            _, c_, h_, w_ = f.shape
            new_features.append(f.view(b, t, c_, h_, w_))

        out = self.middle(new_features, of)
        out = self.decoder(*out)
        out = self.reconstruction_blocks(out)
        out = self.upsample(out)
        return out

    def _single_frame_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for single frame without optical flow

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, 3, H, W)
        """
        self.check_input_shape(x)

        features = self.encoder(x)
        out = self.decoder(*features)

        out = self.reconstruction_blocks(out)
        out = self.upsample(out)
        return out

    def forward(self, x: dict | torch.Tensor) -> torch.Tensor:
        """Forward method

        Args:
            x (dict | torch.Tensor): Input tensor of shape (B, C, H, W) or dictionary containing keys "LQ" and "OF" with values being torch tensors of low quality frames and optical flow of shape (B, T, C, H, W), (B, T - 1, 2, H, W) respectively

        Returns:
            torch.Tensor: Output tensor of shape (B, 3, H, W)
        """

        if self.use_skip_connections:
            return self._multi_frame_of_forward(x)
        else:
            return self._single_frame_forward(x)


class UnetDecoderWithFirstSkip(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError("Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(n_blocks, len(decoder_channels)))

        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
