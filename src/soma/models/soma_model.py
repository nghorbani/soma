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

from collections import OrderedDict

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.labels_map import general_labels_map
from soma.models.model_components import LayeredResConv1d
from soma.models.model_components import Transpose, ResConv1DBlock, SDivide
from soma.models.optimal_transport import log_optimal_transport
from soma.models.transformer import LayeredSelfAttention


def masked_mean(tensor, mask, dim, keepdim=False):
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim)  # Find the average!


class ScorePredictor(nn.Module):
    def __init__(self, num_labels: int, enable_transformer: bool = True, enable_sinkhorn: bool = False,
                 num_total_attention_feat: int = 125, num_attention_layers: int = 8, num_attention_heads: int = 5):
        super(ScorePredictor, self).__init__()

        self.enable_transformer = enable_transformer

        # the multiplication with num_total_attention_feat is to increase number of parameters in case of avoiding transformer
        self.score_predictor_b1 = nn.Sequential(  # per body part
            Transpose(-2, -1),
            ResConv1DBlock(3, num_total_attention_feat),
            # if enable_transformer else ResConv1DBlock(3, 4 * num_total_attention_feat),
            nn.ReLU(), )

        if enable_transformer:
            self.point_attention = LayeredSelfAttention(num_total_attention_feat,
                                                        num_attention_layers,
                                                        num_attention_heads)
        else:
            self.conv1d_block = LayeredResConv1d(num_total_attention_feat,
                                                 num_total_attention_feat,
                                                 num_layers=num_attention_layers)  # the factors is to increase the count of model parameters

        self.score_predictor_b2 = nn.Sequential(nn.ReLU(),
                                                ResConv1DBlock(num_total_attention_feat, num_total_attention_feat * 2),
                                                nn.ReLU(),
                                                SDivide(num_total_attention_feat ** .5),
                                                nn.Conv1d(num_total_attention_feat * 2,
                                                          num_labels if enable_sinkhorn else num_labels + 1, 1, 1),
                                                Transpose(-2, -1),
                                                )

    def forward(self, pts_centered):
        output_dict = {}

        b1_res = self.score_predictor_b1(pts_centered)

        if self.enable_transformer:
            att_res, att_weights = self.point_attention(b1_res)
            output_dict['scores'] = self.score_predictor_b2(att_res)

            output_dict['attention_weights'] = att_weights
        else:
            conv_res = self.conv1d_block(b1_res)
            output_dict['scores'] = self.score_predictor_b2(conv_res)

        return output_dict


class SOMA(nn.Module):

    def __init__(self, cfg: DictConfig):
        super(SOMA, self).__init__()

        superset_fname = cfg.data_parms.marker_dataset.superset_fname

        superset_meta = marker_layout_load(superset_fname, labels_map=general_labels_map)
        num_labels = OrderedDict({k: np.sum(v) for k, v in superset_meta['marker_type_mask'].items()})

        num_total_attention_feat = cfg.model_parms.labeler.num_total_attention_feat
        num_attention_layers = cfg.model_parms.labeler.num_attention_layers
        num_attention_heads = cfg.model_parms.labeler.num_attention_heads

        self.has_multiple_body_parts = len(num_labels) > 1  # num_labels is a dictionary
        self.num_labels = num_labels
        self.enable_transformer = cfg.model_parms.labeler.enable_transformer
        self.enable_sinkhorn = cfg.model_parms.labeler.enable_sinkhorn

        if self.has_multiple_body_parts:
            raise NotImplementedError('This functionality is not released for current SOMA.')

        else:
            num_part_labels = num_labels[list(num_labels.keys())[0]]
            self.score_predictor = ScorePredictor(num_labels=num_part_labels,
                                                  enable_transformer=self.enable_transformer,
                                                  enable_sinkhorn=self.enable_sinkhorn,
                                                  num_total_attention_feat=num_total_attention_feat,
                                                  num_attention_layers=num_attention_layers,
                                                  num_attention_heads=num_attention_heads)

        if self.enable_sinkhorn:
            bin_score = torch.nn.Parameter(torch.tensor(1.))
            self.register_parameter('bin_score', bin_score)
            self.num_sinkhorn_iters = cfg.model_parms.labeler.num_sinkhorn_iters
        else:
            self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, pts):
        """

        Args:
            pts: bs x n_pts x 3

        Returns:

        """
        output_dict = {}

        pts_offset = SOMA.compute_offsets(pts)
        pts_centered = pts - pts_offset

        if self.has_multiple_body_parts:
            raise NotImplementedError('This functionality is not released for current SOMA.')

        else:
            score_predictor_res = self.score_predictor(pts_centered)
            scores = score_predictor_res['scores']
            if 'attention_weights' in score_predictor_res:
                output_dict.update({'attention_weights': score_predictor_res['attention_weights']})

        if self.enable_sinkhorn:

            aug_asmat = log_optimal_transport(scores, self.bin_score, iters=self.num_sinkhorn_iters)

            output_dict.update({
                'label_ids': aug_asmat[:, :-1].argmax(-1),
                'label_confidence': aug_asmat[:, :-1].exp(),
                'aug_asmat': aug_asmat,
            })
        else:
            asmat = self.log_softmax(scores)
            output_dict.update({
                'label_ids': asmat.argmax(-1),
                'label_confidence': asmat.exp(),
                'aug_asmat': asmat,
            })

        return output_dict

    @staticmethod
    def compute_offsets(points):
        """
        given a batch of seq of points compute the center of the points at first time frame
        this is basically th bos offset
        Args:
            points: Nxnum_pointsx3

        Returns:
            Nx1x3

        """
        bs, num_markers, _ = points.shape

        nonzero_mask = ((points == 0.0).sum(-1) != 3)
        offsets = []
        for i in range(bs):
            if nonzero_mask[i].sum() == 0:
                offsets.append(points.new(np.zeros([1,3])))
                continue
            offsets.append(torch.median(points[i, nonzero_mask[i]], dim=0, keepdim=True).values)
        return torch.cat(offsets, dim=0).view(bs, 1, 3)
