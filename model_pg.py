#! python3
# -*- coding: utf-8 -*-
"""
################################################################################################
Implementation of 'PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION'##
https://arxiv.org/pdf/1710.10196.pdf                                                          ##
################################################################################################
https://github.com/shanexn
Created: 2018-06-11
################################################################################################
"""
import os

os.environ["PATH"] = os.environ["PATH"] + ";" + r"E:\dev\python36\for_pytorch\Library\bin"

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, calculate_gain


#################################################################################
# Construct Help Functions Class#################################################
#################################################################################
class HelpFunc(object):
    @staticmethod
    def process_transition(a, b):
        """
        Transit tensor a as tensor b's size by
        'nearest neighbor filtering' and 'average pooling' respectively
        which mentioned below Figure2 of the Paper https://arxiv.org/pdf/1710.10196.pdf
        :param torch.Tensor a: is a tensor with size [batch, channel, height, width]
        :param torch.Tensor b: similar as a
        :return torch.Tensor :
        """
        a_batch, a_channel, a_height, a_width = a.size()
        b_batch, b_channel, b_height, b_width = b.size()
        # Drop feature maps
        if a_channel > b_channel:
            a = a[:, :b_channel]

        if a_height > b_height:
            assert a_height % b_height == 0 and a_width % b_width == 0
            assert a_height / b_height == a_width / b_width
            ks = int(a_height // b_height)
            a = F.avg_pool2d(a, kernel_size=ks, stride=ks, padding=0, ceil_mode=False, count_include_pad=False)

        if a_height < b_height:
            assert b_height % a_height == 0 and b_width % a_width == 0
            assert b_height / a_height == b_width / a_width
            sf = b_height // a_height
            a = F.upsample(a, scale_factor=sf, mode='nearest')

        # Add feature maps.
        if a_channel < b_channel:
            z = torch.zeros((a_batch, b_channel - a_channel, b_height, b_width))
            a = torch.cat([a, z], 1)
        # print("a size: ", a.size())
        return a


#################################################################################
# Construct Generator and Discriminator #########################################
#################################################################################
class Generator(nn.Module):
    def __init__(self,
                 resolution,  # Output resolution. Overridden based on dataset.
                 latent_size,  # Dimensionality of the latent vectors.
                 final_channel=3,  # Output channel size, for rgb always 3
                 fmap_base=2 ** 13,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 9,  # Maximum number of feature maps in any layer.
                 is_tanh=False
                 ):
        super(Generator, self).__init__()
        self.latent_size_ = latent_size
        self.is_tanh_ = is_tanh
        self.final_channel_ = final_channel
        # Use (fmap_max, fmap_decay, fmap_max)
        # to control every level's in||out channels
        self.fmap_base_ = fmap_base
        self.fmap_decay_ = fmap_decay
        self.fmap_max_ = fmap_max
        image_pyramid_ = int(np.log2(resolution))  # max level of the Image Pyramid
        self.resolution_ = 2 ** image_pyramid_  # correct resolution
        self.net_level_max_ = image_pyramid_ - 1  # minus 1 in order to exclude last rgb layer

        self.lod_layers_ = nn.ModuleList()    # layer blocks exclude to_rgb layer
        self.rgb_layers_ = nn.ModuleList()    # rgb layers each correspond to specific level.

        for level in range(self.net_level_max_):
            self._construct_by_level(level)

        self.net_level_ = self.net_level_max_  # set default net level as max level
        self.net_status_ = "stable"            # "stable" or "fadein"
        self.net_alpha_ = 1.0                  # the previous stage's weight

    @property
    def net_config(self):
        """
        Return current net's config.
        The config is used to control forward
        The pipeline was mentioned below Figure2 of the Paper
        """
        return self.net_level_, self.net_status_, self.net_alpha_

    @net_config.setter
    def net_config(self, config_list):
        """
        :param iterable config_list: [net_level, net_status, net_alpha]
        :return:
        """
        self.net_level_, self.net_status_, self.net_alpha_ = config_list

    def forward(self, x):
        """
        The pipeline was mentioned below Figure2 of the Paper
        """
        if self.net_status_ == "stable":
            cur_output_level = self.net_level_
            for cursor in range(self.net_level_+1):
                x = self.lod_layers_[cursor](x)
            x = self.rgb_layers_[cur_output_level](x)

        elif self.net_status_ == "fadein":
            pre_output_level = self.net_level_ - 1
            cur_output_level = self.net_level_
            pre_weight, cur_weight = self.net_alpha_, 1.0 - self.net_alpha_
            output_cache = []
            for cursor in range(self.net_level_+1):
                x = self.lod_layers_[cursor](x)
                if cursor == pre_output_level:
                    output_cache.append(self.rgb_layers_[cursor](x))
                if cursor == cur_output_level:
                    output_cache.append(self.rgb_layers_[cursor](x))
            x = HelpFunc.process_transition(output_cache[0], output_cache[1]) * pre_weight \
                + output_cache[1] * cur_weight

        else:
            raise AttributeError("Please set the net_status: ['stable', 'fadein']")

        return x

    def _construct_by_level(self, cursor):
        in_level = cursor
        out_level = cursor + 1
        in_channels, out_channels = map(self._get_channel_by_stage, (in_level, out_level))
        block_type = "First" if cursor == 0 else "UpSample"
        self._create_block(in_channels, out_channels, block_type)  # construct previous (max_level - 1) layers
        self._create_block(out_channels, 3, "ToRGB")                # construct rgb layer for each previous level

    def _create_block(self, in_channels, out_channels, block_type):
        """
        Create a network block
        :param block_type:  only can be "First"||"UpSample"||"ToRGB"
        :return:
        """
        block_cache = []
        if block_type in ["First", "UpSample"]:
            if block_type == "First":
                block_cache.append(PixelWiseNormLayer())
                block_cache.append(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=4, stride=1, padding=3, bias=False))
            if block_type == "UpSample":
                block_cache.append(nn.Upsample(scale_factor=2, mode='nearest'))
                block_cache.append(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, stride=1, padding=1, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(PixelWiseNormLayer())
            block_cache.append(nn.Conv2d(out_channels, out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(PixelWiseNormLayer())
            self.lod_layers_.append(nn.Sequential(*block_cache))
        elif block_type == "ToRGB":
            block_cache.append(nn.Conv2d(in_channels, out_channels=1,
                                         kernel_size=1, stride=1, padding=0, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            if self.is_tanh_ is True:
                block_cache.append(nn.Tanh())
            self.rgb_layers_.append(nn.Sequential(*block_cache))
        else:
            raise TypeError("'block_type' must in ['First', 'UpSample', 'ToRGB']")

    def _get_channel_by_stage(self, level):
        return min(int(self.fmap_base_ / (2.0 ** (level * self.fmap_decay_))), self.fmap_max_)


class Discriminator(nn.Module):
    def __init__(self,
                 resolution,  # Output resolution. Overridden based on dataset.
                 input_channel=3,  # input channel size, for rgb always 3
                 fmap_base=2 ** 13,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 9,  # Maximum number of feature maps in any layer.
                 is_sigmoid=False
                 ):
        super(Discriminator, self).__init__()
        self.input_channel_ = input_channel
        self.is_sigmoid_ = is_sigmoid
        # Use (fmap_max, fmap_decay, fmap_max)
        # to control every level's in||out channels
        self.fmap_base_ = fmap_base
        self.fmap_decay_ = fmap_decay
        self.fmap_max_ = fmap_max
        image_pyramid_ = int(np.log2(resolution))  # max level of the Image Pyramid
        self.resolution_ = 2 ** image_pyramid_  # correct resolution
        self.net_level_max_ = image_pyramid_ - 1  # minus 1 in order to exclude first rgb layer

        self.lod_layers_ = nn.ModuleList()  # layer blocks exclude to_rgb layer
        self.rgb_layers_ = nn.ModuleList()  # rgb layers each correspond to specific level.

        for level in range(self.net_level_max_, 0, -1):
            self._construct_by_level(level)

        self.net_level_ = self.net_level_max_  # set default net level as max level
        self.net_status_ = "stable"  # "stable" or "fadein"
        self.net_alpha_ = 1.0  # the previous stage's weight

    @property
    def net_config(self):
        return self.net_level_, self.net_status_, self.net_alpha_

    @net_config.setter
    def net_config(self, config_list):
        self.net_level_, self.net_status_, self.net_alpha_ = config_list

    def forward(self, x):
        if self.net_status_ == "stable":
            cur_input_level = self.net_level_max_ - self.net_level_ - 1
            x = self.rgb_layers_[cur_input_level](x)
            for cursor in range(cur_input_level, self.net_level_max_):
                x = self.lod_layers_[cursor](x)

        elif self.net_status_ == "fadein":
            pre_input_level = self.net_level_max_ - self.net_level_
            cur_input_level = self.net_level_max_ - self.net_level_ - 1
            pre_weight, cur_weight = self.net_alpha_, 1.0 - self.net_alpha_
            x_pre_cache = self.rgb_layers_[pre_input_level](x)
            x_cur_cache = self.rgb_layers_[cur_input_level](x)
            x_cur_cache = self.lod_layers_[cur_input_level](x_cur_cache)
            x = HelpFunc.process_transition(x_pre_cache, x_cur_cache) * pre_weight + x_cur_cache * cur_weight

            for cursor in range(cur_input_level + 1, self.net_level_max_):
                x = self.lod_layers_[cursor](x)

        else:
            raise AttributeError("Please set the net_status: ['stable', 'fadein']")

        return x

    def _construct_by_level(self, cursor):
        in_level = cursor
        out_level = cursor - 1
        in_channels, out_channels = map(self._get_channel_by_stage, (in_level, out_level))
        block_type = "Minibatch" if cursor == 1 else "DownSample"
        self._create_block(in_channels, out_channels, block_type)  # construct (max_level-1) layers(exclude rgb layer)
        self._create_block(1, in_channels, "FromRGB")  # construct rgb layer for each previous level

    def _create_block(self, in_channels, out_channels, block_type):
        """
        Create a network block
        :param block_type:  only can be "Minibatch"||"DownSample"||"FromRGB"
        :return:
        """
        block_cache = []
        if block_type == "DownSample":
            block_cache.append(nn.Conv2d(in_channels, out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(nn.Conv2d(out_channels, out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False))
            self.lod_layers_.append(nn.Sequential(*block_cache))
        elif block_type == "FromRGB":
            block_cache.append(nn.Conv2d(in_channels=1, out_channels=out_channels,
                                         kernel_size=1, stride=1, padding=0, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            self.rgb_layers_.append(nn.Sequential(*block_cache))
        elif block_type == "Minibatch":
            block_cache.append(MiniBatchAverageLayer())
            block_cache.append(nn.Conv2d(in_channels + 1, out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(nn.Conv2d(out_channels, out_channels,
                                         kernel_size=4, stride=1, padding=0, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            block_cache.append(nn.LeakyReLU(negative_slope=0.2))
            block_cache.append(nn.Conv2d(out_channels, out_channels=1,
                                         kernel_size=1, stride=1, padding=0, bias=False))
            block_cache.append(EqualizedLearningRateLayer(block_cache[-1]))
            if self.is_sigmoid_ is True:
                block_cache.append(nn.Sigmoid())
            self.lod_layers_.append(nn.Sequential(*block_cache))
        else:
            raise TypeError("'block_type' must in ['Minibatch', 'DownSample', 'FromRGB']")

    def _get_channel_by_stage(self, level):
        return min(int(self.fmap_base_ / (2.0 ** (level * self.fmap_decay_))), self.fmap_max_)
