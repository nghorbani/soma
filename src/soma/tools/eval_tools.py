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
import pandas as pd
from human_body_prior.tools.omni_tools import flatten_list
from loguru import logger
from pandas import ExcelWriter
from sklearn.metrics import classification_report, confusion_matrix

from moshpp.tools.mocap_interface import MocapSession


def find_corresponding_labels(markers_gt: np.ndarray,
                              labels_gt: np.ndarray,
                              markers_rec: np.ndarray,
                              labels_rec: np.ndarray,
                              flatten_output=True,
                              rtol=1e-3, atol=1e-8) -> dict:
    assert len(markers_gt) == len(markers_rec)

    labels_gt = np.array(labels_gt)
    labels_rec = np.array(labels_rec)

    gt_nonan_mask = MocapSession.marker_availability_mask(markers_gt)
    rec_nonan_mask = MocapSession.marker_availability_mask(markers_rec)

    mocap_length = len(markers_gt)

    labels_gt_aligned = []
    rec_labels_aligned = []
    markers_aligned = []

    for t in range(mocap_length):

        # rec (soma) markers must be subset of gt markers. find subset of gt markers assignable to soma markers
        # per frame labels are repeats of trajectory labels. so this will not hold
        # assert (labels_rec[t, rec_nan_mask[t]] == 'nan').sum() == rec_nan_mask[t].sum()

        if not rec_nonan_mask[t].sum() <= gt_nonan_mask[t].sum():
            missing_gt_markers = list(set(labels_rec[t].tolist()).difference(set(labels_gt[t].tolist())))
            if not (len(missing_gt_markers) == 1 and missing_gt_markers[0] == 'nan'):
                logger.info(f'Frame {t}: There are more soma markers than gt markers.'
                            f'f{rec_nonan_mask[t].sum()} vs.'
                            f'f{gt_nonan_mask[t].sum()}.'
                            f' Probably following markers are not labeled in gt mocap: {missing_gt_markers}')

        gt2rec_dist = np.sqrt(
            np.power(markers_rec[t, rec_nonan_mask[t]][None] - markers_gt[t, gt_nonan_mask[t]][:, None], 2))
        gt2rec_best = np.isclose(gt2rec_dist, 0, rtol=rtol, atol=atol).sum(-1) == 3

        if gt2rec_best.sum() == 0: continue
        # assert gt2rec_best.sum(), ValueError('Not a single soma point could be assigned to a gt point. This cannot be true.')
        # print(gt2rec_best.sum(), gt_nonan_mask[t].sum())
        if gt2rec_best.sum() > gt_nonan_mask[t].sum():
            logger.error(f'Frame {t}: There exists overlapping soma-to-gt assignment:'
                         f' {gt2rec_best.sum()} vs. possible {gt_nonan_mask[t].sum()}')
            logger.error('Try reducing either rtol or atol')

        gt_single_assigned = gt2rec_best.sum(-1) == 1
        gt2rec_ids = gt2rec_best.argmax(-1)[gt_single_assigned]

        assert (np.unique(gt2rec_ids, return_counts=True)[1] > 1).sum() == 0, ValueError(
            'Multiple gt labels could be assigned to a rec label.')

        # count_nans = np.logical_not(gt_nonan_mask[t]).sum() + np.logical_not(rec_nonan_mask[t]).sum()
        # rec_labels_aligned.append(labels_rec[t, rec_nonan_mask[t]][gt2rec_ids].tolist() + ['nan' for _ in range(count_nans)])
        # labels_gt_aligned.append(labels_gt[t,gt_nonan_mask[t]][gt_single_assigned].tolist() + ['nan' for _ in range(count_nans)])

        rec_labels_aligned.append(labels_rec[t, rec_nonan_mask[t]][gt2rec_ids].tolist())
        labels_gt_aligned.append(labels_gt[t, gt_nonan_mask[t]][gt_single_assigned].tolist())
        markers_aligned.append(markers_gt[t, gt_nonan_mask[t]][gt_single_assigned].tolist())

    if flatten_output:
        labels_gt_aligned = flatten_list(labels_gt_aligned)
        rec_labels_aligned = flatten_list(rec_labels_aligned)
        markers_aligned = flatten_list(markers_aligned)

    return {'labels_gt': labels_gt_aligned,
            'labels_rec': rec_labels_aligned,
            'markers': markers_aligned}


def compute_labeling_metrics(labels_gt, labels_rec, create_excel_dfs=True, out_fname=None):
    # assert avg_mode in ['micro', 'macro', 'weighted']
    if len(labels_rec) == 0:
        logger.error('No label ever detected by SOMA. have you run the soma_processor?')
        return {'f1': 0,
                'acc': 0,
                'prec': 0,
                'recall': 0}

    # superset_labels = sorted(list(set(labels_gt)))
    superset_labels = sorted(list(set(labels_rec + labels_gt)))
    # if 'nan' not in superset_labels:
    #     superset_labels += ['nan']
    # else:
    #     superset_labels.pop(superset_labels.index('nan'))
    #     superset_labels += ['nan']

    all_label_map = {k: superset_labels.index(k) for k in superset_labels}
    assert len(all_label_map) == len(set(all_label_map.keys()))  # keys should be unique
    #
    label_ids_gt = np.array([all_label_map[k] for k in labels_gt])
    label_ids_rec = np.array([all_label_map[k] for k in labels_rec])

    avg_mode = 'macro'

    # The support is the number of occurrences of each class in y_true.
    # so if a label is not present in the labels_gt but present in labels_rec it will get a 0 percent.
    # this could happen when a maker layout is changed for a capture and soma still assigns a nearby label.
    labeling_report = classification_report(label_ids_gt, label_ids_rec,
                                            output_dict=True, labels=np.arange(len(superset_labels)),
                                            target_names=superset_labels, zero_division=0)

    # accuracy = accuracy_score(label_ids_gt, label_ids_rec)
    # accuracy = jaccard_score(label_ids_gt, label_ids_rec, labels=np.arange(len(superset_labels)), average='macro')
    #
    f1_score = labeling_report[f'{avg_mode} avg']['f1-score']
    precision = labeling_report[f'{avg_mode} avg']['precision']
    recall = labeling_report[f'{avg_mode} avg']['recall']
    accuracy = labeling_report['accuracy']

    results = {'f1': f1_score,
               'acc': accuracy,
               'prec': precision,
               'recall': recall}

    if create_excel_dfs:
        cm = confusion_matrix(label_ids_gt, label_ids_rec, labels=range(len(superset_labels)))

        # per_class_acc = cm.diagonal()/cm.sum(axis=1)
        # for k, v in zip(superset_labels, per_class_acc):
        #     labeling_report[k].update({'acc':v})

        df_cm = pd.DataFrame(cm, index=superset_labels, columns=superset_labels)

        labeling_report = pd.DataFrame(labeling_report).transpose()

        excel_dfs = {'labeling_report': labeling_report,
                     'confusion_matrix': df_cm}
        results.update(excel_dfs)

        if out_fname:
            assert out_fname.endswith('.xlsx')
            save_xlsx(excel_dfs, xlsx_fname=out_fname)

    return results


def save_xlsx(dicts_dfs, xlsx_fname):
    with ExcelWriter(xlsx_fname, engine='xlsxwriter') as writer:
        for name, df in dicts_dfs.items():
            df.to_excel(writer, sheet_name=name)
