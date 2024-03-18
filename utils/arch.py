from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock, DecoderBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from mmcv.ops import DeformConv2d
from typing import List, Optional
import inspect


def make_conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_scales=3):
    convs = []
    for _ in range(num_scales):
        convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        convs.append(nn.ReLU(inplace=True))
    return nn.Sequential(*convs)


def make_layer(layer, num_layers, *args, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(layer(*args, **kwargs))
    return nn.Sequential(*layers)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, num_features):
        self.conv1 = nn.Conv2d(num_features, num_features, 1, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, 1, 1, 1)

    def forward(self, x):
        i = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return F.relu(out + i)


class DConvMotionCompensation(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=3, deformable_groups=1):
        super(DConvMotionCompensation, self).__init__()
        self.deformable_groups = deformable_groups
        self.dconv = DeformConv2d(in_channels, out_channels, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.convs = make_conv_relu(in_channels, in_channels, num_scales=num_scales)
        self.offset_conv = nn.Conv2d(2, 3 * 3 * self.deformable_groups * 2, 1, 1)

    def forward(self, x: torch.Tensor, of: torch.Tensor):
        x = self.convs(x)
        of = self.offset_conv(of)
        out = self.dconv(x, of)
        return out


class MotionCompensationFeatureFusion(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_scales=3, center_frame_idx=1):
        super(MotionCompensationFeatureFusion, self).__init__()
        self.ref_convs = make_conv_relu(in_channels, in_channels, num_scales=num_scales)
        self.end_ref_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.neighboring_blocks = DConvMotionCompensation(in_channels, mid_channels, num_scales=num_scales)
        self.center_frame_idx = center_frame_idx

        self.fusion_block = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels * 3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 3, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, of: torch.Tensor):
        ref = x[:, self.center_frame_idx, ...]
        ref_out = self.end_ref_conv(self.ref_convs(ref))

        neighbor_out = []
        neighbor_indices = [i for i in range(x.size(1)) if i != self.center_frame_idx]
        of_indices = [i for i in range(of.size(1))]
        for (
            ni,
            oi,
        ) in zip(neighbor_indices, of_indices):
            neighbor_out.append(self.neighboring_blocks(x[:, ni, ...], of[:, oi, ...]))

        neighbor_out = torch.cat(neighbor_out, dim=1)

        return self.fusion_block(torch.cat([ref_out, neighbor_out], dim=1))


class PyramidMotionCompensationFetaureFusion(nn.Module):
    def __init__(self, in_channels, mid_channels_scale, out_channels, num_scales=3, center_frame_idx=1, use_previous=False):
        super(PyramidMotionCompensationFetaureFusion, self).__init__()
        self.skip_connections = nn.ModuleDict()
        for level, (ic, oc) in enumerate(zip(in_channels, out_channels)):
            level_name = f"level_{level}"
            self.skip_connections[level_name] = MotionCompensationFeatureFusion(
                ic, ic * mid_channels_scale, oc, num_scales=num_scales, center_frame_idx=center_frame_idx
            )

    def forward(self, xs: List[torch.Tensor], of: torch.Tensor):
        """Forward method

        Args:
            xs (List[torch.Tensor]): List of input feature maps in shape (B, T, C, H, W)
            of [torch.Tensor]: Tensor of optical flow in shape (B, T, 2, H, W)

        Returns:
            _type_: _description_
        """
        self.ofs = []
        b, t, c, h, w = of.shape
        for _, _, _, h_, w_ in [x.shape for x in xs]:
            of_ = F.interpolate(of.view(b * t, c, h, w), size=(h_, w_), mode="bilinear", align_corners=False)
            self.ofs.append(of_.view(b, t, c, h_, w_))

        out = []
        for i, (x, of) in enumerate(zip(xs, self.ofs)):
            level_name = f"level_{i}"
            out.append(self.skip_connections[level_name](x, of))

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
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
    def __init__(self, num_reconstruction_blocks: Optional[int] = None, *args, **kwargs):
        super(SuperResolutionUnet, self).__init__(*args, **kwargs)

        sig = inspect.signature(smp.Unet.__init__)
        default_values = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
        out_channels = default_values["decoder_channels"][-1]

        self.middle = PyramidMotionCompensationFetaureFusion(
            self.encoder.out_channels, 2, self.encoder.out_channels, num_scales=3, center_frame_idx=1
        )
        self.segmentation_head = None
        self.decoder = UnetDecoderWithFirstSkip(
            self.encoder.out_channels,
            default_values["decoder_channels"],
            n_blocks=default_values["encoder_depth"],
            use_batchnorm=default_values["decoder_use_batchnorm"],
            center=True if default_values["encoder_name"].startswith("vgg") else False,
            attention_type=default_values["decoder_attention_type"],
        )
        if num_reconstruction_blocks is not None:
            self.reconstruction_blocks = make_layer(BottleneckResidualBlock, num_reconstruction_blocks, out_channels)
        else:
            self.reconstruction_blocks = nn.Identity()

        self.upsample = UpsampleBlock(out_channels)

    def forward(self, x: dict):
        feat = x["LQ"]
        of = x["OF"]

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
