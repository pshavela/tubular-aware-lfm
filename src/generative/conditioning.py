# Adapted from https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/controlnet.py
# with the following license:
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
import torch.utils.checkpoint
from torch import nn
from monai.networks.blocks import Convolution
from monai.networks.nets.diffusion_model_unet import zero_module
from collections.abc import Sequence


class ConditioningEmbedding(nn.Module):
    """
    Network to encode the semantic label map into a latent embedding.
    Convolutional layers and SILU activations.

    Args:
        spatial_dims: number of spatial dimensions, either 2 or 3.
        in_channels: number of input channels, ie. number of semantic label classes.
        num_channels: number of channels at each level, input will be downsampled with convolutions by a factor
            of 2^(|num_channels| - 1).
        kernel_size: size of convolutional kernels.
        zero_out: Whether to use zero conv final layer.
        use_checkpointing: determines whether to checkpoint the forward pass output, can minimize GPU memory
            usage at the cost of recomputations.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: Sequence[int] = (8, 16, 32),
        kernel_size: int = 3,
        zero_out: bool = False,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=kernel_size,
            padding=1,
            adn_ordering="A",
            act='SWISH',
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(num_channels) - 1):
            channel_in = num_channels[i]
            channel_out = num_channels[i + 1]

            # space-preserving convolution
            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_in,
                    strides=1,
                    kernel_size=kernel_size,
                    padding=1,
                    adn_ordering="A",
                    act="SWISH",
                )
            )

            # down convolution
            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_out,
                    strides=2,
                    kernel_size=kernel_size,
                    padding=1,
                    adn_ordering="A",
                    act="SWISH",
                )
            )

        self.conv_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            out_channels=num_channels[-1],
            strides=1,
            kernel_size=kernel_size,
            padding=1,
            conv_only=True,
        )

        if zero_out:
            self.conv_out = zero_module(self.conv_out)

    def _foward(self, conditioning):
        embedding = self.conv_in(conditioning)

        for block in self.blocks:
            embedding = block(embedding)

        embedding = self.conv_out(embedding)

        return embedding

    def forward(self, conditioning):
        """
        Create embeddings for conditioning.

        Args:
            conditioning: shape BCHW[D], where C = in_channels.

        Returns:
            conditioning embedding.
        """
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self._foward, conditioning, use_reentrant=False)
        return self._foward(conditioning)
