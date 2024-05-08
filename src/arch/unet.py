# Author: Matej Sirovatka
# Adapted from: https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders/unet

import segmentation_models_pytorch as smp
import torch
import inspect
from typing import Optional
import torch.nn as nn
from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock, DecoderBlock

from arch.bottleneck_residual import BottleneckResidualBlock
from arch.pyramid_feature_fusion import PyramidMotionCompensatedFeatureFusion
from arch.upsample import UpsampleBlock
from arch.utils import make_layer


class SRUnetDecoder(nn.Module):
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
        learnable_of: bool = False,
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
            self.middle = PyramidMotionCompensatedFeatureFusion(
                in_channels=self.encoder.out_channels,
                mid_channels_scale=mid_conv_channels_scale,
                out_channels=self.encoder.out_channels,
                num_scales=num_scales,
                center_frame_idx=1,
                use_convs=use_convs,
                use_middle_frame=use_middle_frame,
                use_previous=use_previous,
                learnable_of=learnable_of,
            )

        self.segmentation_head = None
        self.decoder = SRUnetDecoder(
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

    def forward(self, x) -> torch.Tensor:
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
