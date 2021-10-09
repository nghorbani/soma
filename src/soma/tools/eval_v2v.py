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
import pickle

import numpy as np
import seaborn as sns
import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import get_support_data_dir
from human_body_prior.tools.omni_tools import rm_spaces
from omegaconf import OmegaConf, DictConfig

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers

sns.set_theme()

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import makepath

from moshpp.mosh_head import MoSh
from loguru import logger


def prepare_eval_v2v_cfg(*args, **kwargs) -> DictConfig:
    setup_mosh_omegaconf_resolvers()
    if not OmegaConf.has_resolver('resolve_mosh_basename'):
        OmegaConf.register_new_resolver('resolve_mosh_basename',
                                        lambda mocap_fname: rm_spaces(
                                            '.'.join(mocap_fname.split('/')[-1].split('.')[:-1])).replace('_stageii',
                                                                                                          ''))

    app_support_dir = get_support_data_dir(__file__)
    base_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/eval_v2v.yaml'))

    override_cfg_dotlist = [f'{k}={v}' for k, v in kwargs.items()]
    override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

    return OmegaConf.merge(base_cfg, override_cfg)


def produce_body_from_mosh_pkl(mosh_stageii_pkl_fname, support_base_dir):
    mosh_result = MoSh.load_as_amass_npz(mosh_stageii_pkl_fname,
                                         include_markers=True)

    surface_model_type = mosh_result['surface_model_type']
    gender = mosh_result['gender']
    surface_model_fname = osp.join(support_base_dir, surface_model_type, gender, 'model.npz')
    assert osp.exists(surface_model_fname), FileNotFoundError(surface_model_fname)

    if 'num_dmpls' in mosh_result:
        dmpl_fname = osp.join(support_base_dir, surface_model_type, gender, 'dmpl.npz')
        assert osp.exists(dmpl_fname)
    else:
        dmpl_fname = None

    # Todo add object model here
    sm = BodyModel(bm_fname=surface_model_fname,
                   num_betas=mosh_result.get('num_betas', 10),
                   num_expressions=mosh_result.get('num_expressions', 0),
                   num_dmpls=mosh_result.get('num_dmpls', None),
                   dmpl_fname=dmpl_fname)

    mosh_result['surface_f'] = c2c(sm.f)

    time_length = len(mosh_result['trans'])
    selected_frames = range(0, time_length)

    assert time_length == len(mosh_result['trans']), \
        ValueError(
            f'All mosh sequences should have same length. {mosh_stageii_pkl_fname} '
            f'has {len(mosh_result["trans"])} != {time_length}')

    if 'betas' in mosh_result:
        mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=time_length, axis=0)

    body_keys = ['betas', 'trans', 'pose_body', 'root_orient', 'pose_hand']

    if 'v_template' in mosh_result:
        mosh_result['v_template'] = np.repeat(mosh_result['v_template'][None], repeats=time_length, axis=0)
        body_keys += ['v_template']
    if surface_model_type == 'smplx':
        body_keys += ['expression']
    if 'num_dmpls' in mosh_result:
        body_keys += ['dmpls']

    surface_parms = {k: torch.Tensor(v[selected_frames]) for k, v in mosh_result.items() if k in body_keys}

    mosh_result['surface_v'] = c2c(sm(**surface_parms).v)

    return mosh_result


def evaluate_v2v_once(cfg):
    import pickle
    from soma.tools.eval_v2v import produce_body_from_mosh_pkl

    from human_body_prior.tools.omni_tools import makepath

    from os import path as osp

    from loguru import logger
    from soma.tools.eval_v2v import prepare_eval_v2v_cfg

    import numpy as np

    cfg = prepare_eval_v2v_cfg(**cfg)

    logger.info('Begin v2v evaluation.')
    logger.info(f'mosh_gt.stageii_fname: {cfg.mosh_gt.stageii_fname}')
    logger.info(f'mosh_rec.stageii_fname: {cfg.mosh_rec.stageii_fname}')

    pkl_fname = cfg.dirs.eval_pkl_out_fname
    if osp.exists(pkl_fname):
        logger.info(f'V2V evaluation already exists: {pkl_fname}')
        # Todo show metrics
        return

    assert osp.exists(cfg.mosh_gt.stageii_fname), FileNotFoundError(f'{cfg.mosh_gt.stageii_fname}')
    assert osp.exists(cfg.mosh_rec.stageii_fname), FileNotFoundError(f'{cfg.mosh_rec.stageii_fname}')

    body_rec = produce_body_from_mosh_pkl(cfg.mosh_rec.stageii_fname, support_base_dir=cfg.dirs.support_base_dir)

    body_gt = produce_body_from_mosh_pkl(cfg.mosh_gt.stageii_fname, support_base_dir=cfg.dirs.support_base_dir)

    assert body_gt['surface_v'].shape[0] == body_rec['surface_v'].shape[0]

    perframe_v2v = np.sqrt(np.power(body_gt['surface_v'] - body_rec['surface_v'], 2).sum(-1))

    perseq_v2v = {
        'v2v_mean': np.mean(perframe_v2v),
        'v2v_std': np.std(perframe_v2v),
        'v2v_median': np.median(perframe_v2v),
    }

    stats_text = ', '.join([f'{k}={perseq_v2v[k] * 100:.2f}' for k in sorted(perseq_v2v)])
    logger.success(f'{cfg.mosh_rec.ds_name} -- {cfg.mosh_rec.subject_name} -- {cfg.mosh_rec.basename} -- {stats_text}')

    pickle.dump({'res_perframe': perframe_v2v,
                 'res_perseq': perseq_v2v},
                open(makepath(pkl_fname, isfile=True), 'wb'))
    logger.info('created: {}'.format(pkl_fname))


def aggregate_v2v_perframe_results(eval_v2v_aggregate, soma_work_base_dir):
    import pandas as pd
    from soma.tools.eval_tools import save_xlsx
    from collections import OrderedDict
    import sys
    import os
    for aggregate_key, individual_res_fnames in eval_v2v_aggregate.items():
        if len(individual_res_fnames) == 0:
            logger.error(f'no eval_v2v_result found for {aggregate_key}')
            continue
        xlsx_out_fname = makepath(soma_work_base_dir, 'evaluations', f'{aggregate_key}_v2v.xlsx', isfile=True)
        if osp.exists(xlsx_out_fname):
            logger.info(f'aggregared v2v evaluation already exists: {xlsx_out_fname}')
            continue

        log_format = f"{aggregate_key} -- {{message}}"

        logger.remove()

        logger.add(xlsx_out_fname.replace('.xlsx', '.log'), format=log_format, enqueue=True)
        logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

        num_motions = len(individual_res_fnames)

        results_perframe = OrderedDict()
        results_perseq = OrderedDict()

        num_frames = 0
        for fname_count, eval_label_individual_res_fname in enumerate(individual_res_fnames):
            subject_name, motion_basename = eval_label_individual_res_fname.split('/')[-2:]
            motion_id = f'{subject_name}/{motion_basename}'.replace('_v2v.', '.')
            try:
                seq_res = pickle.load(open(eval_label_individual_res_fname, 'rb'))
            except Exception as e:
                logger.error(f'Issue with {eval_label_individual_res_fname}: {e} -- removing it.')
                os.remove(eval_label_individual_res_fname)

            results_perseq[motion_id] = seq_res['res_perseq']
            results_perframe[motion_id] = seq_res['res_perframe']
            num_frames += len(seq_res['res_perframe'])

            stats_text = ', '.join(
                [f'{k}={seq_res["res_perseq"][k] * 1000:.2f}' for k in sorted(seq_res['res_perseq'])])
            logger.success(f'{fname_count + 1}/{num_motions} -- {stats_text}')

        subject_names = list(set([k.split('/')[0] for k in results_perframe.keys()]))
        motion_names = list(set([k.split('/')[1].split('_')[0] for k in results_perframe.keys()]))

        all_v2v_frames = np.concatenate(list(results_perframe.values()), 0)
        aggregated_stats = {
            'v2v_std': np.nanstd(all_v2v_frames),
            'v2v_mean': np.nanmean(all_v2v_frames),
            'v2v_median': np.nanmedian(all_v2v_frames),
            'num_frames': num_frames,
            'num_motions': num_motions,
            'num_subject': len(subject_names)
        }

        results_persubject = {}
        for subject_name in subject_names:
            persubject_motions = [v for k, v in results_perframe.items() if k.startswith(subject_name)]
            persubject_v2v_frames = np.concatenate(persubject_motions, 0)
            results_persubject[subject_name] = {
                'v2v_std': np.nanstd(persubject_v2v_frames),
                'v2v_mean': np.nanmean(persubject_v2v_frames),
                'v2v_median': np.nanmedian(persubject_v2v_frames),
                'num_frames': persubject_v2v_frames.shape[0],
                'num_motions': len(persubject_motions),
            }

        results_permotion = {}
        for motion_name in motion_names:
            permotion_motions = [v for k, v in results_perframe.items() if motion_name in k]
            permotion_v2v_frames = np.concatenate(permotion_motions, 0)
            results_permotion[motion_name] = {
                'v2v_std': np.nanstd(permotion_v2v_frames),
                'v2v_mean': np.nanmean(permotion_v2v_frames),
                'v2v_median': np.nanmedian(permotion_v2v_frames),
                'num_frames': permotion_v2v_frames.shape[0],
                'num_motions': len(permotion_motions),
            }

        stats_all_text = ', '.join(
            [f'{k}={aggregated_stats[k] * 1000:.2f}' if k.startswith('v2v') else f'{k}={aggregated_stats[k]:}' for k in
             sorted(aggregated_stats)])
        logger.success(f'{stats_all_text}')

        excel_dfs = {
            'results_perseq': pd.DataFrame(results_perseq).transpose(),
            'results_persubject': pd.DataFrame(results_persubject).transpose(),
            'results_permotion': pd.DataFrame(results_permotion).transpose(),

            'aggregated_stats': pd.DataFrame(aggregated_stats, index=['Total']).transpose(),
        }

        save_xlsx(excel_dfs, xlsx_fname=xlsx_out_fname)
        logger.info(f'created: {xlsx_out_fname}')
