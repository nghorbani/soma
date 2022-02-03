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

from human_body_prior.tools.omni_tools import get_support_data_dir
from loguru import logger
from omegaconf import OmegaConf

from moshpp.mosh_head import MoSh
from moshpp.mosh_head import run_moshpp_once
from moshpp.tools.mocap_interface import MocapSession
from moshpp.tools.run_tools import universal_mosh_jobs_filter
from soma.render.blender_tools import prepare_render_cfg  # could be commented if no rendering is expected
from soma.render.blender_tools import render_mosh_once
from soma.tools.parallel_tools import run_parallel_jobs


def mosh_manual(
        mocap_fnames,
        mosh_cfg=None,
        render_cfg=None,
        parallel_cfg=None,
        # mosh_stagei_perseq = False,
        **kwargs):
    '''

    :param mocap_fnames: list
    :param mosh_cfg: dict
    :param render_cfg:dict
    :param parallel_cfg:
    # :param mosh_stagei_perseq: dict
    :param kwargs: bool
    :return:
    '''
    if parallel_cfg is None: parallel_cfg = {}
    if mosh_cfg is None: mosh_cfg = {}
    if render_cfg is None: render_cfg = {}

    run_tasks = kwargs.get('run_tasks', ['mosh', 'render'])

    only_stagei = kwargs.get('only_stagei', False)
    fast_dev_run = kwargs.get('fast_dev_run', False)
    determine_shape_for_each_seq = kwargs.get('determine_shape_for_each_seq', False)

    app_support_dir = get_support_data_dir(__file__)

    fname_filter = kwargs.get('fname_filter', None)

    mosh_jobs = []
    render_jobs = []
    exclude_mosh_job_keys = []

    if fast_dev_run: mocap_fnames = mocap_fnames[:3]

    for mocap_fname in mocap_fnames:

        if fname_filter:
            if not sum([i in mocap_fname for i in fname_filter]): continue
        mocap_key = '_'.join(mocap_fname.split('/')[-3:-1])

        persubject_marker_layout = kwargs.get('persubject_marker_layout', False)
        mosh_job = mosh_cfg.copy()
        mosh_job.update({
            'mocap.fname': mocap_fname,
        })
        if persubject_marker_layout:
            # todo: do we need to pick the mocaps to produce the layout here?
            mosh_job.update({'dirs.marker_layout.fname':
                                 '${dirs.work_base_dir}/${mocap.ds_name}/${mocap.ds_name}_${mocap.session_name}_${surface_model.type}.json',
                             })
        # if mosh_stagei_perseq:
        #     mosh_job['dirs.stagei_fname'] = \
        #         '${dirs.work_base_dir}/${mocap.ds_name}/${mocap.session_name}/${mocap.basename}_stagei.pkl'
        #     mosh_job['moshpp.stagei_frame_picker.stagei_mocap_fnames'] = [mocap_fname]
        #     # mosh_job['dirs.marker_layout.fname'] = \
        #     #     '${dirs.work_base_dir}/${mocap.ds_name}/${mocap.session_name}/${mocap.basename}_${surface_model.type}.json'

        cur_mosh_cfg = MoSh.prepare_cfg(**mosh_job.copy())
        perseq_mosh_stagei = cur_mosh_cfg.moshpp.perseq_mosh_stagei

        try:
            cur_mosh_cfg.dirs.stagei_fname
        except Exception as e:
            mocap_subject_mask = MocapSession(mocap_fname, 'mm').subject_mask
            logger.error(e)
            raise ValueError("Could not find the correct multisubject gender settings.json. "
                             "Specify the interested subject and the desired model gender: subjects: #markers {} \n "
                             "A sample settings file {} at the session folder would be: {} \n".format(
                {sname: smask.sum() for sname, smask in mocap_subject_mask.items()},
                osp.join(osp.dirname(mocap_fname), 'settings.json'),
                {sname: {"gender": "unknown"} for sname in mocap_subject_mask.keys()}))
        render_key = '_'.join(mocap_fname.split('/')[-3:])
        if cur_mosh_cfg.mocap.subject_id >= 0 and cur_mosh_cfg.mocap.subject_name not in cur_mosh_cfg.mocap.subject_names:
            logger.warning(f'subject name {cur_mosh_cfg.mocap.subject_name} '
                           f'not available in mocap subjects {cur_mosh_cfg.mocap.subject_names} of {mocap_fname}')
            continue

        mocap_subjects = cur_mosh_cfg.mocap.subject_names if cur_mosh_cfg.mocap.subject_id >= 0 else [None]

        for subject_id, subject_name in enumerate(mocap_subjects):
            if subject_name is not None:

                if cur_mosh_cfg.mocap.multi_subject and subject_name == 'null': continue
                mocap_key += f'_{subject_name}'

                cur_mosh_cfg.mocap.subject_id = subject_id
                # in case subject name is given it wont change after subject_id change
                if cur_mosh_cfg.mocap.subject_name != subject_name: continue
                mosh_job.update({'mocap.subject_id': subject_id})

            if 'mosh-stagei' in run_tasks:
                mosh_job.update({'runtime.stagei_only': True})
                if osp.exists(cur_mosh_cfg.dirs.stagei_fname): continue

            if not osp.exists(cur_mosh_cfg.dirs.stageii_fname):
                mosh_jobs.append(mosh_job.copy())
                continue  # mosh results are not available

            if render_key not in exclude_render_job_keys:
                render_job = render_cfg.copy()
                if cur_mosh_cfg.mocap.multi_subject:
                    stageii_fname_split = cur_mosh_cfg.dirs.stageii_fname.split('/')
                    stageii_fname_split[-2] = '*'
                    mosh_stageii_fnames = glob('/'.join(stageii_fname_split))
                else:
                    mosh_stageii_fnames = [cur_mosh_cfg.dirs.stageii_fname]

                render_job.update({
                    'mesh.mosh_stageii_pkl_fnames': mosh_stageii_fnames,
                })
                cur_render_cfg = prepare_render_cfg(**render_job)
                if not osp.exists(cur_render_cfg.dirs.mp4_out_fname):
                    render_jobs.append(render_job.copy())
                    exclude_render_job_keys.append(render_key)

    if np.any([task in run_tasks for task in ['mosh', 'mosh-stagei']]):
        logger.info('Submitting MoSh++ jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/moshpp_parallel.yaml'))
        moshpp_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        mosh_jobs = universal_mosh_jobs_filter(mosh_jobs, determine_shape_for_each_seq=perseq_mosh_stagei)

        run_parallel_jobs(func=run_moshpp_once, jobs=mosh_jobs, parallel_cfg=moshpp_parallel_cfg)

    if 'render' in run_tasks:
        logger.info('Submitting render jobs.')

        base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/blender_parallel.yaml'))
        render_parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))
        run_parallel_jobs(func=render_mosh_once, jobs=render_jobs, parallel_cfg=render_parallel_cfg)
    return len(render_jobs) > 0 and len(mosh_jobs) > 0
