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

"""
This script runs MoSh on subset datasets of AMASS to prepare SOMA's train and validation body parameters and simulated markers.
Simulated markers can be used to prepare AMASS marker noise dataset.
If you dont want to run this script you can acquire only the AMASS SMPL-X gender neutral body parameters from amass.is.tue.mpg.de.
We release AMASS marker noise model for pretrained SOMA models.
"""

import os.path as osp

from human_body_prior.tools.omni_tools import get_support_data_dir
from loguru import logger
from omegaconf import OmegaConf

from moshpp.mosh_head import MoSh
from moshpp.mosh_head import run_moshpp_once
from soma.amass.amass_info import amass_datasets
from soma.render.blender_tools import prepare_render_cfg
from soma.render.blender_tools import render_mosh_once
from soma.tools.parallel_tools import run_parallel_jobs


def prepare_amass_smplx(mosh_cfg: dict = None,
                        render_cfg: dict = None,
                        parallel_cfg: dict = None,
                        **kwargs):
    if parallel_cfg is None: parallel_cfg = {}
    if mosh_cfg is None: mosh_cfg = {}
    if render_cfg is None: render_cfg = {}

    run_tasks = kwargs.get('run_tasks', ['mosh', 'render'])

    only_stagei = kwargs.get('only_stagei', False)
    only_datasets = kwargs.get('only_datasets', None)
    fast_dev_run = kwargs.get('fast_dev_run', False)
    determine_shape_for_each_seq = kwargs.get('determine_shape_for_each_seq', False)

    app_support_dir = get_support_data_dir(__file__)

    fname_filter = kwargs.get('fname_filter', None)

    mosh_jobs = []
    render_jobs = []
    exclude_mosh_job_keys = []

    for ds_name, ds_cfg in amass_datasets.items():
        if (only_datasets is not None) and (ds_name not in only_datasets): continue

        assert len(ds_cfg['mocap_fnames']) > 0, ValueError(f'Found no mocap for {ds_name}')
        logger.info(f"Found #{len(ds_cfg['mocap_fnames'])} mocaps for {ds_name}")

        if fast_dev_run: ds_cfg['mocap_fnames'] = ds_cfg['mocap_fnames'][:3]

        for mocap_fname in ds_cfg['mocap_fnames']:

            if fname_filter:
                if not sum([i in mocap_fname for i in fname_filter]): continue
            mocap_key = '_'.join(mocap_fname.split('/')[-3:-1])

            persubject_marker_layout = ds_cfg.get('persubject_marker_layout', False)
            mosh_job = mosh_cfg.copy()
            mosh_job.update({
                'mocap.fname': mocap_fname,
                'mocap.ds_name': ds_name,
                **ds_cfg['mosh_cfg_override']
            })
            if persubject_marker_layout:
                # todo: do we need to pick the mocaps to produce the layout here?
                mosh_job.update({
                    'dirs.marker_layout_fname': '${dirs.work_base_dir}/${mocap.ds_name}/${mocap.ds_name}_${mocap.subject_name}_${surface_model.type}.json',
                })
            if 'subject_specific_settings' in ds_cfg:
                subject_name = mocap_fname.split('/')[-2]
                if subject_name in ds_cfg['subject_specific_settings']:
                    mosh_job.update(**ds_cfg['subject_specific_settings'][subject_name])

            cur_mosh_cfg = MoSh.prepare_cfg(**mosh_job.copy())

            if only_stagei and osp.exists(cur_mosh_cfg.dirs.stagei_fname): continue

            if mocap_key not in exclude_mosh_job_keys and not osp.exists(cur_mosh_cfg.dirs.stageii_fname):
                if not osp.exists(cur_mosh_cfg.dirs.stagei_fname) and not determine_shape_for_each_seq: exclude_mosh_job_keys.append(mocap_key)
                mosh_jobs.append(mosh_job.copy())
                continue

            if osp.exists(cur_mosh_cfg.dirs.stageii_fname):
                render_job = render_cfg.copy()
                render_job.update({
                    'mesh.mosh_stageii_pkl_fnames': [cur_mosh_cfg.dirs.stageii_fname],
                    **ds_cfg.get('render_cfg_override', {})
                })
                cur_render_cfg = prepare_render_cfg(**render_job)
                if not osp.exists(cur_render_cfg.dirs.mp4_out_fname):
                    render_jobs.append(render_job)

    if 'mosh' in run_tasks:
        logger.info('Submitting MoSh++ jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/moshpp_parallel.yaml'))
        moshpp_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        run_parallel_jobs(func=run_moshpp_once, jobs=mosh_jobs, parallel_cfg=moshpp_parallel_cfg)

    if 'render' in run_tasks:
        logger.info('Submitting render jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/blender_parallel.yaml'))
        render_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        run_parallel_jobs(func=render_mosh_once, jobs=render_jobs, parallel_cfg=render_parallel_cfg)


if __name__ == '__main__':
    # only_datasets = ['HumanEva', 'ACCAD', 'TotalCapture', 'CMU', 'Transitions', 'PosePrior']
    # only_datasets = ['SSM']
    # only_datasets = list(set(amass_datasets.keys()).difference(set(only_datasets)))
    prepare_amass_smplx(
        mosh_cfg={
            'moshpp.verbosity': 1,
            'surface_model.gender': 'neutral',
            'dirs.work_base_dir': '/is/cluster/scratch/nghorbani/amass/mosh_results/20210726/amass_neutral',
            # 'dirs.work_base_dir': '/is/cluster/scratch/nghorbani/amass/mosh_results/20210726/amass_gender_specific',
        },
        render_cfg={
            'dirs.work_base_dir': '/is/cluster/scratch/nghorbani/amass/mp4_renders/20210726/amass_neutral',
            # 'dirs.work_base_dir': '/is/cluster/scratch/nghorbani/amass/mp4_renders/20210726/amass_gender_specific',
            'render.render_engine': 'cycles',  # eevee / cycles,
            # 'render.render_engine': 'cycles',  # eevee / cycles,
            # 'render.save_final_blend_file': True
            'render.floor.enable': False,
        },
        parallel_cfg={
            'pool_size': 0,
            'max_num_jobs': -1,
            'randomly_run_jobs': True,
        },
        run_tasks=[
            'mosh',
            'render',
        ],
        # fast_dev_run=True,
        # only_datasets=[
        #     'WEIZMANN',
        #                ],
        # only_datasets=only_datasets,
        # only_datasets=['SSM', 'HumanEva', 'ACCAD', 'TotalCapture', 'CMU', 'Transitions', 'PosePrior', 'HDM05'],
        # fname_filter=['29/29_12', '86/86_05', '86/86_04'],  # ['SSM_synced/resynced/20160330_03333'],
    )
