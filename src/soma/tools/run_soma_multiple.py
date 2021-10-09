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
import os.path
import os.path as osp
from glob import glob
from typing import List

import numpy as np
from human_body_prior.tools.omni_tools import get_support_data_dir
from human_body_prior.tools.omni_tools import rm_spaces
from loguru import logger
from omegaconf import OmegaConf

from moshpp.mosh_head import MoSh
from moshpp.mosh_head import run_moshpp_once
from moshpp.tools.run_tools import universal_mosh_jobs_filter
from soma.render.blender_tools import prepare_render_cfg
from soma.render.blender_tools import render_mosh_once
from soma.tools.eval_labeling import evaluate_labeling_once, aggregate_labeling_perframe_results
from soma.tools.eval_labeling import prepare_eval_label_cfg
from soma.tools.eval_v2v import evaluate_v2v_once, aggregate_v2v_perframe_results
from soma.tools.eval_v2v import prepare_eval_v2v_cfg
from soma.tools.parallel_tools import run_parallel_jobs
from soma.tools.soma_processor import SOMAMoCapPointCloudLabeler
from soma.tools.soma_processor import run_soma_once


def run_soma_on_multiple_settings(soma_expr_ids: List[str], soma_mocap_target_ds_names: List[str],
                                  soma_data_ids: List[str] = None, tracklet_labeling_options: List[bool] = None,
                                  ds_name_gt: str = None, soma_cfg: dict = None, mosh_cfg: dict = None,
                                  render_cfg: dict = None, eval_label_cfg: dict = None, parallel_cfg: dict = None,
                                  eval_v2v_cfg: dict=None,
                                  fast_dev_run: bool = False,
                                  run_tasks: List[str] = None,
                                  mocap_ext: str = '.c3d',
                                  mosh_stagei_perseq: bool = False,
                                  fname_filter: List[str] = None,
                                  soma_work_base_dir: str = None,
                                  mocap_base_dir: str = None, gt_mosh_base_dir: str = None,
                                  **kwargs):
    """
    Run multiple SOMA models on various settings
    Args:
        soma_expr_ids: list of soma experiment ids
        soma_mocap_target_ds_names: target dataset names, these should be available at mocap_base_dir
        soma_data_ids: data ids of some experiments
        tracklet_labeling_options: whether to use tracklet labeling
        ds_name_gt: gt mocap data for labeling evaluation
        soma_cfg: overloading soma_run_conf.yaml
        mosh_cfg: overloading moshpp_conf.yaml inside mosh code
        render_cfg: overload render_conf.yaml
        eval_label_cfg: eval_label.yaml
        parallel_cfg: relevant for use on IS cluster
        fast_dev_run: if True will run for a limited number of mocaps
        run_tasks: a selection of ['soma', 'mosh', 'render', 'eval_label']
        mocap_ext: file extension of the source mocap point clouds
        mosh_stagei_perseq: if True stage-i of mosh will run for every sequence instead of every subject
        fname_filter: List of strings to filter the source mocaps
        mocap_base_dir: base directory for source mocaps
        gt_mosh_base_dir: directory holding mosh results of the gt mocaps, used for v2v evaluation
        soma_work_base_dir: base directory for soma data. this directory holds: data, training_experiments, support_data
        **kwargs:


    Returns:

    """
    app_support_dir = get_support_data_dir(__file__)

    if run_tasks is None:
        run_tasks = ['soma',  # run auto-labeling of SOMA on mocap point cloud
                     'mosh',  # solve autolabeled mocaps using MoSh
                     'render',  # render the solved bodies with blender
                     'eval_label',  # evaluation labeling
                     'eval_label_aggregate',  # aggregate labeling metrics
                     'eval_v2v',  # evaluate surface reconstruction as v2v euclidean distance
                     'eval_v2v_aggregate'  # aggregate v2v results
                     ]

    if soma_data_ids is None: soma_data_ids = ['OC_05_G_03_real_000_synt_100']
    if tracklet_labeling_options is None: tracklet_labeling_options = [True]

    if soma_cfg is None: soma_cfg = {}
    if mosh_cfg is None: mosh_cfg = {}
    if render_cfg is None: render_cfg = {}
    if eval_label_cfg is None: eval_label_cfg = {}
    if eval_v2v_cfg is None: eval_v2v_cfg = {}
    if parallel_cfg is None: parallel_cfg = {}

    if mocap_base_dir is None:
        mocap_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'

    if soma_work_base_dir is None:
        soma_work_base_dir = '/is/cluster/scratch/nghorbani/soma'

    soma_jobs = []
    mosh_jobs = []
    render_jobs = []
    eval_label_jobs = []
    eval_label_aggregate = {}
    eval_v2v_jobs = []
    eval_v2v_aggregate = {}
    for soma_mocap_target_ds_name in soma_mocap_target_ds_names:
        mocap_fnames = glob(osp.join(mocap_base_dir, soma_mocap_target_ds_name, f'*/*{mocap_ext}'))
        if fast_dev_run: mocap_fnames = mocap_fnames[:5]
        for soma_expr_id in soma_expr_ids:
            for soma_data_id in soma_data_ids:
                for tracklet_labeling in tracklet_labeling_options:
                    eval_aggregate_key = f'{soma_mocap_target_ds_name}_{soma_expr_id}_{soma_data_id}' + (
                        '_tracklet' if tracklet_labeling else '')
                    eval_label_aggregate[eval_aggregate_key] = []
                    eval_v2v_aggregate[eval_aggregate_key] = []
                    for mocap_fname in mocap_fnames:
                        assert mocap_fname == rm_spaces(mocap_fname), ValueError(
                            f'mocap_fname has space in the text: {mocap_fname}')
                        if fname_filter:
                            if not sum([i in mocap_fname for i in fname_filter]): continue
                        soma_job = soma_cfg.copy()
                        soma_job.update({
                            'mocap.fname': mocap_fname,
                            'soma.expr_id': soma_expr_id,
                            'soma.data_id': soma_data_id,
                            'dirs.work_base_dir': soma_work_base_dir,
                            'soma.tracklet_labeling.enable': tracklet_labeling
                        })
                        cur_soma_cfg = SOMAMoCapPointCloudLabeler.prepare_cfg(**soma_job)
                        assert osp.exists(cur_soma_cfg.soma.expr_dir), FileExistsError(cur_soma_cfg.soma.expr_dir)
                        soma_labeled_mocap_fname = cur_soma_cfg.dirs.mocap_out_fname
                        if not os.path.exists(soma_labeled_mocap_fname):
                            soma_jobs.append(soma_job)
                            continue  # soma results are not available
                        if run_tasks == ['soma']: continue
                        eval_label_job = eval_label_cfg.copy()

                        eval_label_job.update({
                            'mocap_gt.fname': mocap_fname,
                            'mocap_gt.unit': cur_soma_cfg.mocap.unit,
                            'mocap_rec.fname': cur_soma_cfg.dirs.mocap_out_fname,
                            'dirs.work_base_dir': cur_soma_cfg.dirs.work_dir.replace('_labeled_mocap', '_eval'),

                            # 'soma.expr_id': soma_expr_id,
                            # 'soma.data_id': soma_data_id,
                            # 'soma.tracklet_labeling.enable': tracklet_labeling
                        })

                        if ds_name_gt:
                            gt_mocap_fname = osp.join(mocap_base_dir, ds_name_gt,
                                                      cur_soma_cfg.mocap.subject_name,
                                                      f'{cur_soma_cfg.mocap.basename}{mocap_ext}')
                            assert osp.exists(gt_mocap_fname), FileNotFoundError(f'{gt_mocap_fname}')
                            eval_label_job.update({
                                'mocap_gt.fname': gt_mocap_fname,
                            })
                        cur_eval_label_cfg = prepare_eval_label_cfg(**eval_label_job)
                        if not os.path.exists(cur_eval_label_cfg.dirs.eval_pkl_out_fname):
                            eval_label_jobs.append(eval_label_job)
                        else:
                            if 'eval_label_aggregate' in run_tasks:
                                eval_label_aggregate[eval_aggregate_key].append(
                                    cur_eval_label_cfg.dirs.eval_pkl_out_fname)

                        if np.all(sorted(run_tasks) == sorted(['soma', 'eval_label'])): continue

                        mosh_job = mosh_cfg.copy()
                        mosh_job.update({
                            'mocap.fname': soma_labeled_mocap_fname,
                            'mocap.unit': 'm',  # soma pkl files are in meters
                            'dirs.work_base_dir': cur_soma_cfg.dirs.work_dir.replace('soma_labeled_mocap',
                                                                                     'mosh_results'),
                        })
                        if mosh_stagei_perseq:
                            mosh_job['dirs.stagei_fname'] = \
                                '${dirs.work_base_dir}/${mocap.ds_name}/${mocap.subject_name}/${mocap.basename}_stagei.pkl'
                            mosh_job['moshpp.stagei_frame_picker.stagei_mocap_fnames'] = [soma_labeled_mocap_fname]
                            mosh_job[
                                'dirs.marker_layout_fname'] = '${dirs.work_base_dir}/${mocap.ds_name}/${mocap.subject_name}/${mocap.basename}_${surface_model.type}.json'

                        cur_mosh_cfg = MoSh.prepare_cfg(**mosh_job.copy())
                        mosh_stageii_pkl_fname = cur_mosh_cfg.dirs.stageii_fname

                        if not osp.exists(mosh_stageii_pkl_fname):
                            mosh_jobs.append(mosh_job)
                            continue  # mosh results are not available

                        render_job = render_cfg.copy()
                        render_job.update({
                            'mesh.mosh_stageii_pkl_fnames': [mosh_stageii_pkl_fname],
                            'dirs.work_base_dir': cur_soma_cfg.dirs.work_dir.replace('soma_labeled_mocap',
                                                                                     'blender_renders'),
                        })
                        cur_render_cfg = prepare_render_cfg(**render_job)
                        if not osp.exists(cur_render_cfg.dirs.mp4_out_fname):
                            render_jobs.append(render_job)

                        if gt_mosh_base_dir:
                            mosh_gt_stageii_fname = osp.join(gt_mosh_base_dir, ds_name_gt,
                                                             cur_soma_cfg.mocap.subject_name,
                                                             f'{cur_mosh_cfg.mocap.basename}_stageii.pkl')
                            if not osp.exists(mosh_gt_stageii_fname):
                                logger.error(f'mosh_gt results do not exist: {mosh_gt_stageii_fname}')
                                continue

                            eval_v2v_job = eval_v2v_cfg.copy()

                            eval_v2v_job.update({
                                'mosh_gt.stageii_fname': mosh_gt_stageii_fname,
                                'mosh_rec.stageii_fname': cur_mosh_cfg.dirs.stageii_fname,
                                'dirs.work_base_dir': cur_soma_cfg.dirs.work_dir.replace('_labeled_mocap', '_eval')
                            })
                            cur_eval_v2v_cfg = prepare_eval_v2v_cfg(**eval_v2v_job)
                            if not os.path.exists(cur_eval_v2v_cfg.dirs.eval_pkl_out_fname):
                                eval_v2v_jobs.append(eval_v2v_job)
                            else:
                                eval_v2v_aggregate[eval_aggregate_key].append(cur_eval_v2v_cfg.dirs.eval_pkl_out_fname)

    if 'soma' in run_tasks:
        logger.info('Submitting SOMA jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/soma_run_parallel.yaml'))
        soma_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        run_parallel_jobs(func=run_soma_once, jobs=soma_jobs, parallel_cfg=soma_parallel_cfg)

    if 'mosh' in run_tasks:
        logger.info('Submitting MoSh++ jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/moshpp_parallel.yaml'))
        moshpp_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        mosh_jobs = universal_mosh_jobs_filter(mosh_jobs, determine_shape_for_each_seq=mosh_stagei_perseq)
        run_parallel_jobs(func=run_moshpp_once, jobs=mosh_jobs, parallel_cfg=moshpp_parallel_cfg)

    if 'render' in run_tasks:
        logger.info('Submitting render jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/blender_parallel.yaml'))
        render_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        run_parallel_jobs(func=render_mosh_once, jobs=render_jobs, parallel_cfg=render_parallel_cfg)

    if 'eval_label' in run_tasks:
        logger.info('Submitting SOMA label evaluations jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/eval_label_parallel.yaml'))
        eval_label_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        run_parallel_jobs(func=evaluate_labeling_once, jobs=eval_label_jobs, parallel_cfg=eval_label_parallel_cfg)

    if len(eval_label_jobs) == 0 and 'eval_label_aggregate' in run_tasks:
        aggregate_labeling_perframe_results(eval_label_aggregate, soma_work_base_dir)

    if 'eval_v2v' in run_tasks:
        logger.info('Submitting SOMA v2v evaluations jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/eval_v2v_parallel.yaml'))
        eval_v2v_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        run_parallel_jobs(func=evaluate_v2v_once, jobs=eval_v2v_jobs, parallel_cfg=eval_v2v_parallel_cfg)

    if len(eval_v2v_jobs) == 0 and 'eval_v2v_aggregate' in run_tasks:
        aggregate_v2v_perframe_results(eval_v2v_aggregate, soma_work_base_dir)
