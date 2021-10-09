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
import torch
from torch import nn


def scaled_dot_product_attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    attention_weight = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', attention_weight, value), attention_weight


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, num_total_attention_feat: int):
        super(MultiHeadedAttention, self).__init__()

        assert num_total_attention_feat % num_heads == 0, ValueError(
            f"num_total_attention_feat ({num_total_attention_feat}) % num_heads ({num_heads}) is not 0 ({num_total_attention_feat % num_heads})")
        self.dim = num_total_attention_feat // num_heads
        self.num_heads = num_heads

        self.proj = nn.ModuleList(
            [nn.Conv1d(num_total_attention_feat, num_total_attention_feat, kernel_size=1) for _ in range(3)])

        self.merge = nn.Conv1d(num_total_attention_feat, num_total_attention_feat, kernel_size=1)

        self.post_merge = nn.Sequential(
            nn.Conv1d(2 * num_total_attention_feat, 2 * num_total_attention_feat, kernel_size=1),
            nn.BatchNorm1d(2 * num_total_attention_feat),
            nn.ReLU(),
            nn.Conv1d(2 * num_total_attention_feat, num_total_attention_feat, kernel_size=1),
        )

        nn.init.constant_(self.post_merge[-1].bias, 0.0)

    def forward(self, init_query, key, value):
        batch_dim = init_query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) for l, x in
                             zip(self.proj, (init_query, key, value))]

        x, attention_weight = scaled_dot_product_attention(query, key, value)

        x = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        x = self.post_merge(torch.cat([x, init_query], dim=1))

        return x, attention_weight


class LayeredSelfAttention(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int, num_attention_heads: int = 4, return_attention_weights=True):
        super(LayeredSelfAttention, self).__init__()
        self.self_attention_layers = nn.ModuleList([
            MultiHeadedAttention(num_attention_heads, feature_dim)
            for _ in range(num_layers)])
        self.return_attention_weights = return_attention_weights

    def forward(self, point_feats):
        """

        Parameters
        ----------
        point_feats: num_batch x num_points x num_feat

        Returns
        -------

        """
        all_attention_weights = []
        for self_attention_layer in self.self_attention_layers:
            new_point_feats, attention_weight = self_attention_layer(point_feats, point_feats, point_feats)
            point_feats = point_feats + new_point_feats
            all_attention_weights.append(attention_weight)

        if self.return_attention_weights:
            return point_feats, torch.stack(all_attention_weights, axis=1)
        else:
            return point_feats
