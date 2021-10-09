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
import os.path as osp
from glob import glob

import numpy as np
from loguru import logger

from soma.amass.mosh_manual import mosh_manual


def gen_stagei_mocap_fnames(mocap_base_dir, subject_name, ext='.c3d'):
    stagei_mocap_fnames = [osp.join(mocap_base_dir, subject_name, frame) for frame in
                           {
                               'soma_subject1': [
                                   f'run_002{ext}_001091',
                                   f'jump_001{ext}_000137',
                                   f'run_001{ext}_001366',
                                   f'jump_001{ext}_000509',
                                   f'throw_001{ext}_000596',
                                   f'dance_003{ext}_001488',
                                   f'jump_001{ext}_000588',
                                   f'squat_002{ext}_001134',
                                   f'jump_002{ext}_000471',
                                   f'run_001{ext}_000032',
                                   f'dance_001{ext}_001042',
                                   f'dance_001{ext}_000289'
                               ],
                               'soma_subject2': [
                                   f'dance_005{ext}_001289',
                                   f'random_004{ext}_000166',
                                   f'run_001{ext}_000826',
                                   f'random_004{ext}_000001',
                                   f'jump_001{ext}_000871',
                                   f'squat_003{ext}_000543',
                                   f'squat_003{ext}_000696',
                                   f'squat_003{ext}_001769',
                                   f'dance_003{ext}_001207',
                                   f'jump_001{ext}_000550',
                                   f'run_001{ext}_000865',
                                   f'throw_001{ext}_000069'
                               ]
                           }[subject_name]]

    available_stagei_mocap_fnames = [osp.exists('_'.join(f.split('_')[:-1])) for f in stagei_mocap_fnames]
    assert sum(available_stagei_mocap_fnames) == len(available_stagei_mocap_fnames), \
        FileNotFoundError(np.array(stagei_mocap_fnames)[np.logical_not(available_stagei_mocap_fnames)])

    return stagei_mocap_fnames