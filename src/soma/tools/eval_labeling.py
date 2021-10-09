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
import os
import pickle
from os import path as osp

import numpy as np
from human_body_prior.tools.omni_tools import get_support_data_dir, makepath
from loguru import logger
from omegaconf import OmegaConf, DictConfig

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers


def prepare_eval_label_cfg(*args, **kwargs) -> DictConfig:
    setup_mosh_omegaconf_resolvers()

    app_support_dir = get_support_data_dir(__file__)
    base_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/eval_label.yaml'))

    override_cfg_dotlist = [f'{k}={v}' for k, v in kwargs.items()]
    override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

    return OmegaConf.merge(base_cfg, override_cfg)


def evaluate_labeling_once(cfg):
    from soma.tools.eval_labeling import prepare_eval_label_cfg
    import pickle

    from soma.tools.eval_tools import find_corresponding_labels

    import numpy as np
    from human_body_prior.tools.omni_tools import makepath
    from moshpp.marker_layout.labels_map import general_labels_map
    from moshpp.tools.mocap_interface import MocapSession

    from soma.tools.eval_tools import compute_labeling_metrics, save_xlsx

    from os import path as osp
    import sys

    from loguru import logger
    from human_body_prior.tools.omni_tools import flatten_list

    cfg = prepare_eval_label_cfg(**cfg)

    logger.remove()

    log_format = f"{{module}}:{{function}}:{{line}} -- {{message}}"
    logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

    logger.info('Beginning to evaluate labeling performance.')
    logger.info(f'mocap_rec: {cfg.mocap_rec.fname}')
    logger.info(f'mocap_gt: {cfg.mocap_gt.fname}')

    # soma_cfg = load_model(expr_dir=cfg.soma.expr_dir, load_only_cfg=True)
    # superset_fname = soma_cfg.data_parms.marker_dataset.superset_fname

    pkl_fname = cfg.dirs.eval_pkl_out_fname
    if osp.exists(pkl_fname):
        logger.info(f'Label evaluation already exists: {pkl_fname}')
        # Todo show metrics
        return

    assert osp.exists(cfg.mocap_rec.fname), FileNotFoundError(f'{cfg.mocap_rec.fname}')
    assert osp.exists(cfg.mocap_gt.fname), FileNotFoundError(f'{cfg.mocap_gt.fname}')

    mocap_rec = MocapSession(mocap_fname=cfg.mocap_rec.fname,
                             mocap_unit=cfg.mocap_rec.unit,
                             mocap_rotate=cfg.mocap_rec.rotate,
                             labels_map=general_labels_map,
                             ignore_stared_labels=False)

    mocap_gt = MocapSession(mocap_fname=cfg.mocap_gt.fname,
                            mocap_unit=cfg.mocap_gt.unit,
                            mocap_rotate=cfg.mocap_gt.rotate,
                            labels_map=general_labels_map,
                            ignore_stared_labels=False)

    T = len(mocap_gt)

    if 'labels_perframe' in mocap_gt._marker_data:
        labels_rec = mocap_rec._marker_data['labels_perframe']
        labels_gt = mocap_gt._marker_data['labels_perframe']
    else:
        labels_rec = np.repeat(np.array(mocap_rec.labels)[None], repeats=T, axis=0)
        labels_gt = np.repeat(np.array(mocap_gt.labels)[None], repeats=T, axis=0)

    assert len(labels_rec) == len(labels_gt)

    if cfg.mocap_gt.set_labels_to_nan:
        logger.debug(f'Setting labels of gt mocap to nan: {cfg.mocap_gt.set_labels_to_nan}')
        for t in range(len(labels_gt)):
            labels_gt[t] = ['nan' if l in cfg.mocap_gt.set_labels_to_nan else l for l in labels_gt[t]]
            # if np.any([l in cfg.mocap_gt.set_labels_to_nan for l in labels_rec[t]]):
            #     logger.error(f'frame {t} has labels that soma has not been trained for.')
            #     raise ValueError

    # soma_superset_labels = list(marker_layout_load(superset_fname,
    #                                                labels_map=general_labels_map)['marker_colors'].keys())

    # rec_unique_labels = np.unique(labels_rec.reshape(-1))
    # gt_unique_labels = np.unique(labels_gt.reshape(-1))
    #
    # soma_labels_not_in_gt = set(rec_unique_labels).difference(set(gt_unique_labels))
    # if len(soma_labels_not_in_gt):
    #     logger.info(
    #         f'It seems that some rec labels are unused and not present in the gt mocap. Will set them to nan: {soma_labels_not_in_gt}')
    #     for t in range(len(labels_rec)):
    #         labels_rec[t] = [l if l in gt_unique_labels else 'nan' for l in labels_rec[t]]
    #
    # gt_labels_not_in_soma = set(gt_unique_labels).difference(set(rec_unique_labels))
    # if len(gt_labels_not_in_soma):
    #     logger.info(
    #         f'It seems that some gt labels are not present in the rec mocap. Will set them to nan: {gt_labels_not_in_soma}')
    #     for t in range(len(labels_gt)):
    #         labels_gt[t] = [l if l in rec_unique_labels else 'nan' for l in labels_gt[t]]

    aligned_res = find_corresponding_labels(mocap_gt.markers,
                                            labels_gt,
                                            mocap_rec.markers,
                                            labels_rec,
                                            flatten_output=False,
                                            rtol=1e-3, atol=1e-8)

    res_perframe = {}
    for t in range(T):
        if len(aligned_res['labels_rec'][t]) == 0 or (aligned_res['labels_rec'][t]) == 0:
            logger.error('A frame of mocap was not assigned to gt data. stopping the evaluation.')
            return
        res = compute_labeling_metrics(aligned_res['labels_gt'][t],
                                       aligned_res['labels_rec'][t],
                                       create_excel_dfs=False)

        for k, v in res.items():
            if k not in res_perframe: res_perframe[k] = []
            res_perframe[k].append(v)

    res_perseq = compute_labeling_metrics(flatten_list(aligned_res['labels_gt']),
                                          flatten_list(aligned_res['labels_rec']),
                                          create_excel_dfs=True)

    for k, v in res_perframe.items():
        res_perseq[f'{k}_std'] = np.nanstd(v)
        res_perseq[f'{k}_mean'] = np.nanmean(v)
        res_perseq[f'{k}_median'] = np.nanmedian(v)

    stats_text = ', '.join([f'{k}={res_perseq[k] * 100:.2f}' for k in sorted(res_perseq) if
                            isinstance(res_perseq[k], float) and '_' not in k])
    logger.success(
        f'{cfg.mocap_rec.ds_name} -- {cfg.mocap_rec.subject_name} -- {cfg.mocap_rec.basename} -- {stats_text}')

    pickle.dump({'res_perframe': res_perframe,
                 'res_perseq': {k: v for k, v in res_perseq.items() if isinstance(v, float)},
                 'labels_gt': aligned_res['labels_gt'],
                 'labels_rec': aligned_res['labels_rec'],
                 'markers': aligned_res['markers']},
                open(makepath(pkl_fname, isfile=True), 'wb'))
    logger.info(f'created: {pkl_fname}')

    excel_dfs = {
        'labeling_report': res_perseq['labeling_report'],
        'confusion_matrix': res_perseq['confusion_matrix'],
    }
    xlsx_fname = pkl_fname.replace('.pkl', '.xlsx')
    save_xlsx(excel_dfs, xlsx_fname=makepath(xlsx_fname, isfile=True))
    logger.info(f'created: {xlsx_fname}')


def aggregate_labeling_perframe_results(eval_label_aggregate, soma_work_base_dir):
    import pandas as pd
    from human_body_prior.tools.omni_tools import flatten_list
    from soma.tools.eval_tools import compute_labeling_metrics, save_xlsx
    from collections import OrderedDict
    import sys

    for aggregate_key, individual_res_fnames in eval_label_aggregate.items():
        if len(individual_res_fnames) == 0:
            logger.error(f'no eval_label_result found for {aggregate_key}')
            continue
        xlsx_out_fname = makepath(soma_work_base_dir, 'evaluations', f'{aggregate_key}_labeling.xlsx', isfile=True)
        if osp.exists(xlsx_out_fname):
            logger.info(f'aggregared labeling evaluation already exists: {xlsx_out_fname}')
            continue

        log_format = f"{aggregate_key} -- {{message}}"

        logger.remove()

        logger.add(xlsx_out_fname.replace('.xlsx', '.log'), format=log_format, enqueue=True)
        logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

        num_motions = len(individual_res_fnames)

        results_perframe = {}
        results_perseq = {}
        labels_soma = []
        labels_gt = []
        res_keys = []
        num_frames = 0
        for fname_count, eval_label_individual_res_fname in enumerate(individual_res_fnames):
            subject_name, motion_basename = eval_label_individual_res_fname.split('/')[-2:]
            motion_id = f'{subject_name}/{motion_basename}'.replace('_labeling.', '.')
            try:
                seq_res = pickle.load(open(eval_label_individual_res_fname, 'rb'))
            except Exception as e:
                logger.error(f'Issue with {eval_label_individual_res_fname}: {e} -- removing it.')
                os.remove(eval_label_individual_res_fname)

            results_perframe[motion_id] = seq_res['res_perframe']
            results_perseq[motion_id] = seq_res['res_perseq']

            res_keys = list(seq_res['res_perframe'].keys())

            labels_soma.extend(flatten_list(seq_res['labels_rec']))
            labels_gt.extend(flatten_list(seq_res['labels_gt']))
            num_frames += len(seq_res['labels_gt'])

            stats_text = ', '.join(
                ['{}={:.2f}'.format(k, seq_res['res_perseq'][k] * 100) for k in sorted(seq_res['res_perseq']) if
                 isinstance(seq_res['res_perseq'][k], float)])
            logger.success(f'{fname_count + 1}/{num_motions} -- {stats_text}')

        subject_names = list(set([k.split('/')[0] for k in results_perframe.keys()]))
        motion_names = list(set([k.split('/')[1].split('_')[0] for k in results_perframe.keys()]))

        aggregated_stats = OrderedDict()
        for res_key in sorted(res_keys):
            all_res_frames = np.concatenate([v[res_key] for v in results_perframe.values()], 0)

            aggregated_stats[f'{res_key}_std'] = np.nanstd(all_res_frames)
            aggregated_stats[f'{res_key}_mean'] = np.nanmean(all_res_frames)
            aggregated_stats[f'{res_key}_median'] = np.nanmedian(all_res_frames)
        aggregated_stats.update({'num_motions': num_motions,
                                 'num_subject': len(subject_names)})

        results_persubject = OrderedDict()
        for subject_name in subject_names:
            results_persubject[subject_name] = OrderedDict()
            persubject_motions = [v for k, v in results_perframe.items() if k.startswith(subject_name)]

            for res_key in sorted(res_keys):
                persubject_frames = np.concatenate([v[res_key] for v in persubject_motions], 0)

                results_persubject[subject_name].update({
                    f'{res_key}_std': np.nanstd(persubject_frames),
                    f'{res_key}_mean': np.nanmean(persubject_frames),
                    f'{res_key}_median': np.nanmedian(persubject_frames),
                })
            results_persubject[subject_name].update({'num_frames': persubject_frames.shape[0],
                                                     'num_motions': len(persubject_motions)})

        results_permotion = OrderedDict()
        for motion_name in motion_names:
            results_permotion[motion_name] = OrderedDict()

            permotion_motions = [v for k, v in results_perframe.items() if motion_name in k]
            for res_key in sorted(res_keys):
                permotion_frames = np.concatenate([v[res_key] for v in permotion_motions], 0)

                results_permotion[motion_name].update({
                    f'{res_key}_std': np.nanstd(permotion_frames),
                    f'{res_key}_mean': np.nanmean(permotion_frames),
                    f'{res_key}_median': np.nanmedian(permotion_frames),
                })

            results_permotion[motion_name].update({'num_frames': permotion_frames.shape[0],
                                                   'num_motions': len(permotion_motions)})
        stats_text = ', '.join(
            ['{}={:.2f}'.format(k, aggregated_stats[k] * 100) for k in sorted(aggregated_stats) if
             isinstance(aggregated_stats[k], float)])
        logger.success(f'{stats_text} -- #motions: {len(individual_res_fnames)}, #frames {num_frames}')

        overall_res = compute_labeling_metrics(labels_gt, labels_soma, create_excel_dfs=True)

        aggregated_stats['num_frames'] = num_frames
        aggregated_stats['num_motions'] = num_motions

        excel_dfs = {
            'labeling_report': overall_res['labeling_report'],
            'confusion_matrix': overall_res['confusion_matrix'],
            'results_perseq': pd.DataFrame(results_perseq).transpose(),
            'results_persubject': pd.DataFrame(results_persubject).transpose(),
            'results_permotion': pd.DataFrame(results_permotion).transpose(),
            'aggregated_stats': pd.DataFrame(aggregated_stats, index=['Total']).transpose(),
        }

        save_xlsx(excel_dfs, xlsx_fname=xlsx_out_fname)
        logger.info('created: {}'.format(xlsx_out_fname))
