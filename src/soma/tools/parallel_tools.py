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
from pathlib import Path
from typing import List
from typing import Union

from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf


def run_parallel_jobs(func, jobs: List[DictConfig], parallel_cfg: DictConfig = None,
                      base_parallel_cfg: Union[DictConfig, Union[Path, str]] = None) -> None:

    if parallel_cfg is None:
        parallel_cfg = {}  # todo update parallel cfg in case it is provided

    if base_parallel_cfg is None:
        base_parallel_cfg = {}
    elif not isinstance(base_parallel_cfg, DictConfig):
        base_parallel_cfg = OmegaConf.load(base_parallel_cfg)

    parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))

    pool_size = parallel_cfg.pool_size
    logger.info(f'#Job(s) submitted: {len(jobs)}')
    max_num_jobs = parallel_cfg.get('max_num_jobs', -1)
    if max_num_jobs and max_num_jobs > 0:
        jobs = jobs[:max_num_jobs]
        logger.info(f'max_num_jobs is set to {max_num_jobs}. choosing the first #Job(s): {len(jobs)}')

    if pool_size==0:
        raise NotImplementedError('This functionality is not released for current SOMA.')


    if parallel_cfg.randomly_run_jobs:
        from random import shuffle
        shuffle(jobs)
        logger.info(f'Will run the jobs in random order.')
    if len(jobs) == 0: return



    if pool_size == 0:
        raise NotImplementedError('This functionality is not released for current SOMA.')

    elif pool_size < 0:
        return
    else:
        for job in jobs:
            func(job)
            # print(job)
