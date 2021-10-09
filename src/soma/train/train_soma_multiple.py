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
from os import path as osp

from human_body_prior.tools.omni_tools import get_support_data_dir
from typing import List

from soma.tools.parallel_tools import run_parallel_jobs
from soma.train.soma_trainer import SOMATrainer
from soma.train.soma_trainer import train_soma_once


def train_multiple_soma(soma_data_settings: List[tuple]=None, soma_train_cfg: dict = None, parallel_cfg: dict = None):
    '''
    Train multiple soma models with various settings.
    Args:
        soma_data_settings: list of tuples with format
            (number of occlusions, number of ghost points, percentage of real data, percentage of synthetic data)
            the data type percentages would work if corresponding type actually exists.
        soma_train_cfg: a dictionary with keys as dot formatted hierarchy and desired values
        parallel_cfg: relevant when running on IS-Condor cluster

    Returns:

    '''
    if soma_train_cfg is None: soma_train_cfg = {}
    if parallel_cfg is None: parallel_cfg = {}
    if soma_data_settings is None: soma_data_settings = [(5, 3, 0.0, 1.0)]

    app_support_dir = get_support_data_dir(__file__)
    base_parallel_cfg_fname = osp.join(app_support_dir, 'conf/parallel_conf/soma_train_parallel.yaml')


    for num_occ_max, num_ghost_max, limit_real_data, limit_synt_data in soma_data_settings:
        job = {
            'data_parms.mocap_dataset.num_occ_max': num_occ_max,
            'data_parms.mocap_dataset.num_ghost_max': num_ghost_max,
            'data_parms.mocap_dataset.limit_real_data': limit_real_data,
            'data_parms.mocap_dataset.limit_synt_data': limit_synt_data,
        }
        job.update(soma_train_cfg)
        cur_soma_cfg = SOMATrainer.prepare_cfg(**job)
        parallel_cfg['jobs_unique_name'] = f'{cur_soma_cfg.soma.expr_id}_{cur_soma_cfg.soma.data_id}'
        run_parallel_jobs(train_soma_once, [job], parallel_cfg=parallel_cfg, base_parallel_cfg=base_parallel_cfg_fname)
