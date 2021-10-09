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

import numpy as np
import torch


class MANO():
    def __init__(self):
        raise NotImplementedError('This functionality is not released for current SOMA.')


class MANO_Torch():
    def __init__(self):
        raise NotImplementedError('This functionality is not released for current SOMA.')

class VPoser():
    def __init__(self):
        raise NotImplementedError('This functionality is not released for current SOMA.')

def right2left_aangle(right_aangle):
    raise NotImplementedError('This functionality is not released for current SOMA.')


def fullrightpose2leftpose(rightpose):
    raise NotImplementedError('This functionality is not released for current SOMA.')


def hand_pose_sequence_generator(handL_frames, handR_frames, hand_prior_type='mano'):
    raise NotImplementedError('This functionality is not released for current SOMA.')



