# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
# If you use this code in a research publication please cite the following:
#
# @inproceedings{SOMA:ICCV:2021,
#   title = {{SOMA}: Solving Optical MoCap Automatically},
#   author = {Ghorbani, Nima and Black, Michael J.},
#   booktitle = {Proceedings of IEEE/CVF International Conference on Computer Vision (ICCV)},
#   month = oct,
#   year = {2021},
#   doi = {},
#   month_numeric = {10}}
#
# You can find complementary content at the project website: https://soma.is.tue.mpg.de/
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
# While at Max-Planck Institute for Intelligent Systems, Tübingen, Germany
#
# 2021.06.18
from torch import nn


def conv1d_layered(channels: list):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class ResDenseBlock(nn.Module):

    def __init__(self, num_feat_in, num_feat_out, num_h=256):
        super(ResDenseBlock, self).__init__()

        self.res_dense = nn.Sequential(
            nn.Linear(num_feat_in, num_h),
            nn.BatchNorm1d(num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_feat_out),
            nn.BatchNorm1d(num_feat_out)
        )

        self.res_dense_short = nn.Sequential(
            *([nn.Linear(num_feat_in, num_feat_out), nn.BatchNorm1d(num_feat_out)] if num_feat_in != num_feat_out else [
                nn.Identity()])
        )

    def forward(self, x):
        return self.res_dense(x) + self.res_dense_short(x)


class ResConv1DBlock(nn.Module):
    """ a series of 1D convolutions with residuals"""

    def __init__(self, num_feat_in, num_feat_out, num_h=256):
        super(ResConv1DBlock, self).__init__()

        self.res_conv1d = nn.Sequential(
            nn.Conv1d(num_feat_in, num_h, 1, 1),
            nn.BatchNorm1d(num_h),
            nn.ReLU(),
            nn.Conv1d(num_h, num_feat_out, 1, 1),
            nn.BatchNorm1d(num_feat_out)
        )

        self.res_conv1d_short = nn.Sequential(
            *([nn.Conv1d(num_feat_in, num_feat_out, 1, 1), nn.BatchNorm1d(num_feat_out)]
              if num_feat_in != num_feat_out else [nn.Identity()])
        )

    def forward(self, x):
        return self.res_conv1d(x) + self.res_conv1d_short(x)


class LayeredResConv1d(nn.Module):

    def __init__(self, num_feat_in, num_feat_out, num_layers, num_h=256, ):
        super(LayeredResConv1d, self).__init__()

        # self.res_conv1d_blocks = nn.ModuleList([
        #     nn.Sequential(ResConv1DBlock(num_feat_in, num_feat_out * 3, num_h=num_h),
        #                   nn.ReLU(),
        #                   nn.Conv1d(num_feat_out * 3, num_feat_out, kernel_size=1)
        #                   ) for _ in range(num_layers)])
        self.res_conv1d_blocks = nn.ModuleList([
            nn.Sequential(ResConv1DBlock(num_feat_in, num_feat_out * 3),
                          nn.ReLU(),
                          nn.Conv1d(num_feat_out * 3, num_feat_out, kernel_size=1),
                          nn.BatchNorm1d(num_feat_out),
                          nn.ReLU(),
                          nn.Conv1d(num_feat_out, num_feat_out, kernel_size=1),
                          ) for _ in range(num_layers)])

    def forward(self, point_feats):
        """

        Parameters
        ----------
        point_feats: num_batch x num_points x num_feat

        Returns
        -------

        """
        for res_conv1d_block in self.res_conv1d_blocks:
            new_point_feats = res_conv1d_block(point_feats)
            point_feats = point_feats + new_point_feats

        return point_feats


class Contiguous(nn.Module):
    def __init__(self):
        super(Contiguous, self).__init__()
        self._name = 'contiguous'

    def forward(self, x):
        return x.contiguous()


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.shape = args
        self._name = 'permute'

    def forward(self, x):
        return x.permute(self.shape)


class Transpose(nn.Module):
    def __init__(self, *args):
        super(Transpose, self).__init__()
        self.shape = args
        self._name = 'transpose'

    def forward(self, x):
        return x.transpose(*self.shape)


class SDivide(nn.Module):
    def __init__(self, scale):
        super(SDivide, self).__init__()
        self.scale = scale
        self._name = 'scalar_divide'

    def forward(self, x):
        return x / self.scale


class SelectItem(nn.Module):
    # https://stackoverflow.com/a/54660829
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
