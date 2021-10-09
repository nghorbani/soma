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
import fnmatch
import json
import os.path as osp
import pickle
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import make_deterministic
from human_body_prior.tools.omni_tools import makepath
from human_body_prior.tools.omni_tools import rm_spaces, flatten_list
from loguru import logger

from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.labels_map import general_labels_map
from moshpp.tools.mocap_interface import MocapSession
from soma.data.mocap_noise_tools import make_ghost_points, occlude_markers, break_trajectories


def inject_synthetic_noise_into_real_mocap(dataset_mocap_fnames: Dict[str, List[Union[str, Path]]],
                                           marker_layout_fnames: dict,
                                           mocap_out_base_dir: Union[str, Path],
                                           oc_g_bt_settings: List[tuple] = None,
                                           mocap_unit: str = 'mm',
                                           mocap_rotate: list = None,
                                           use_exact_num_oc: bool = False,
                                           fname_filter: List[str] = None,
                                           ) -> None:
    """
    Corrupt a mocap with occlusions and ghost markers as well as broken trajectories

    Args:
        dataset_mocap_fnames: source mocap files of a dataset
        marker_layout_fnames: we can choose which markersets to be used in the corrupted mocap dataset
        mocap_out_base_dir: where to put the corrupted mocaps at
        oc_g_bt_settings: occlusion/ghost points/broken trajectory settings
        mocap_unit: default unit for c3d files is mm
        mocap_rotate: make the z axis up
        use_exact_num_oc: whether the number of occlusions is up-to o is exact
        fname_filter: capability to filter file names
    """
    if oc_g_bt_settings is None:
        oc_g_bt_settings = [(5, 3, 50), ]
    log_format = f"{{module}}:{{function}}:{{line}} -- {{message}}"
    logger.remove()

    for marker_layout_name, marker_layout_fname in marker_layout_fnames.items():

        marker_layout_labels = list(
            marker_layout_load(marker_layout_fname, labels_map=general_labels_map)['marker_vids'].keys())

        for ds_name, mocap_fnames in dataset_mocap_fnames.items():
            make_deterministic(100)

            for num_occ_max, num_ghost_max, num_btraj_max in oc_g_bt_settings:
                new_ds_name = f'{ds_name}___{marker_layout_name}___OC_{num_occ_max:02d}_G_{num_ghost_max:02d}_BT_{num_btraj_max:02d}'
                if use_exact_num_oc:
                    new_ds_name = new_ds_name.replace('___OC_', '___OCE_')
                # if fname_filter is not None:
                #     new_ds_name = new_ds_name + '__filtered'

                new_ds_out_dir = makepath(mocap_out_base_dir, new_ds_name)

                shutil.copy(marker_layout_fname, makepath(new_ds_out_dir, 'marker_layout.json', isfile=True))

                log_fname = makepath(new_ds_out_dir, 'log.log', isfile=True)
                logger.add(log_fname, format=log_format, enqueue=True)
                logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

                logger.debug(f'#{len(mocap_fnames)} source mocaps found for {ds_name}')

                for mocap_fname in sorted(mocap_fnames):
                    if 'stagei.npz' in mocap_fname: continue
                    subject_name, mocap_basename = mocap_fname.split('/')[-2:]
                    mocap_basename = rm_spaces('.'.join(mocap_basename.split('.')[:-1]))

                    if fname_filter is not None:
                        if not np.any([fnmatch.fnmatch(mocap_fname, a) for a in fname_filter]):
                            continue
                    mocap_out_fname = makepath(new_ds_out_dir, subject_name, f'{mocap_basename}.pkl', isfile=True)
                    if osp.exists(mocap_out_fname): continue
                    logger.debug(f'Starting to process {mocap_fname}')

                    mocap_gt = MocapSession(mocap_fname,
                                            mocap_unit=mocap_unit,
                                            mocap_rotate=mocap_rotate,
                                            only_markers=marker_layout_labels,
                                            labels_map=general_labels_map
                                            )

                    if len(mocap_gt.labels) / len(set(mocap_gt.labels)) > 1.5:
                        logger.error(
                            f'Current mocap seems to have multiple subjects: len unique labels {len(mocap_gt.labels)}, len labels {len(set(mocap_gt.labels))} -- {mocap_fname}')
                        continue
                    markers = torch.from_numpy(mocap_gt.markers).type(torch.float32)
                    superset_labels = mocap_gt.labels + ['nan']
                    nan_class_id = superset_labels.index('nan')

                    time_length, num_markers = markers.shape[:-1]
                    if time_length < 3 * num_btraj_max:
                        logger.error(f'Number of frames ({time_length}) is smaller than 3 times '
                                     f'the requested num_btraj_max ({num_btraj_max}) for: {mocap_fname}')
                        continue

                    label_ids = torch.from_numpy(
                        np.repeat(np.arange(len(superset_labels) - 1)[None], axis=0, repeats=time_length)).type(
                        torch.long)

                    if num_occ_max != 0:
                        for t in range(time_length):
                            num_occ = num_occ_max if use_exact_num_oc else np.random.choice(num_occ_max)
                            markers[t] = occlude_markers(markers[t], num_occ)

                        gt_occ_rate = np.logical_not(MocapSession.marker_availability_mask(mocap_gt.markers)).sum() / (
                                time_length * num_markers)
                        cur_occ_rate = np.logical_not(MocapSession.marker_availability_mask(markers)).sum() / (
                                time_length * num_markers)

                        logger.debug(
                            f'Original occ rate {gt_occ_rate * 100:.2f}% and with synthetic occlusion {cur_occ_rate * 100:.2f}%')

                    if num_ghost_max != 0:
                        ghost_markers = make_ghost_points(markers, num_ghost_max=num_ghost_max, use_upto_num_ghost=True)
                        markers = torch.cat([markers, ghost_markers], dim=1)
                        ghost_label_ids = np.array([nan_class_id for _ in range(num_ghost_max)])
                        ghost_label_ids = torch.from_numpy(
                            np.repeat(ghost_label_ids[None], repeats=time_length, axis=0)).type(torch.long)

                        label_ids = torch.cat([label_ids, ghost_label_ids], dim=-1)

                        avail_ghost_markers = MocapSession.marker_availability_mask(ghost_markers)
                        perframe_ghost_rate = avail_ghost_markers.sum() / (time_length * markers.shape[1])
                        logger.debug(f'Average per-frame ghost rate: {perframe_ghost_rate * 100:.2f}%')

                    nan_mask = np.logical_not(MocapSession.marker_availability_mask(markers))
                    label_ids[nan_mask] = nan_class_id

                    # randomly permute markers along trajectories
                    permvecs = np.random.permutation(markers.shape[1])

                    markers = markers[:, permvecs]
                    label_ids = label_ids[:, permvecs]

                    if num_btraj_max != 0:
                        markers, label_ids = break_trajectories(markers, label_ids,
                                                                nan_class_id=nan_class_id,
                                                                num_btraj_max=num_btraj_max)

                    labels_perframe = np.array(superset_labels)[label_ids]
                    pickle.dump({
                        'markers': c2c(markers),
                        'labels': ['*{:03d}'.format(d) for d in range(markers.shape[1])],
                        'labels_perframe': labels_perframe,
                        'frame_rate': mocap_gt.frame_rate
                    }, open(makepath(mocap_out_fname, isfile=True), 'wb'), protocol=2)
                    logger.debug(f'Created {mocap_out_fname}, [{markers.shape}]')

                    gender_out_fname = makepath(new_ds_out_dir, subject_name, 'settings.json', isfile=True)
                    gender_source_fname = osp.join(osp.dirname(mocap_fname), 'settings.json')

                    if not osp.exists(gender_out_fname):
                        if not osp.exists(gender_source_fname):
                            gender_source_fname = gender_source_fname.replace('settings.json', 'gender.json')
                        if not osp.exists(gender_source_fname): continue
                        gender = json.load(open(gender_source_fname))['gender']

                        with open(gender_out_fname, 'w') as f:
                            json.dump({"gender": gender}, f, sort_keys=True, indent=4, separators=(',', ': '))
                logger.remove()


def KIT():
    mocap_out_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'
    mocap_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/original'
    mocap_fnames = glob(osp.join(mocap_dir, 'KIT', '*/*.c3d'))

    args = {
        'marker_layout_fnames': {
            'KIT': '/ps/project/soma/support_files/release_soma/marker_layouts/KIT/423/downstairs01_smplx_finetuned.json', },
        'dataset_mocap_fnames': {'KIT': mocap_fnames},
        'oc_g_bt_settings': [(5, 3, 50), ],
        'mocap_unit': 'mm',
        'mocap_rotate': None,
        'use_exact_num_oc': False,
        # 'fname_filter': ['*/3/*'],
        'mocap_out_base_dir': mocap_out_base_dir,
    }

    inject_synthetic_noise_into_real_mocap(**args)


def BMLrub():  # 120fps
    mocap_out_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'
    mocap_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/original'
    mocap_fnames = glob(osp.join(mocap_dir, 'BMLrub', '*/*.pkl'))

    args = {
        'marker_layout_fnames': {
            'BMLrub': '/ps/project/soma/support_files/release_soma/marker_layouts/BMLrub/rub001/0007_normal_jog1_smplx_finetuned.json'},
        'dataset_mocap_fnames': {'BMLrub': mocap_fnames},
        'oc_g_bt_settings': [(5, 3, 50)],
        'mocap_unit': 'mm',
        'use_exact_num_oc': False,
        'mocap_rotate': None,
        'fname_filter': None,
        'mocap_out_base_dir': mocap_out_base_dir,
    }

    inject_synthetic_noise_into_real_mocap(**args)


def BMLrub_test_ghorbani_permutation_2019():  # 120fps
    mocap_out_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'
    mocap_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/original'
    mocap_fnames = glob(osp.join(mocap_dir, 'BMLrub', '*/*.pkl'))

    fname_filter = flatten_list(
        [[f'*{subject_id:03d}/*{action_name}*' for subject_id in range(68, 92)] for action_name in
         ['sit', 'jump', 'walk', 'jog']])

    args = {
        'marker_layout_fnames': {
            'BMLrub': '/ps/project/soma/support_files/release_soma/marker_layouts/BMLrub/rub001/0007_normal_jog1_smplx_finetuned.json'},

        'dataset_mocap_fnames': {'BMLrub_test_ghorbani_permutation_2019': mocap_fnames},
        'oc_g_bt_settings': [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (5, 3, 0), (0, 0, 0)],
        'mocap_unit': 'mm',
        'use_exact_num_oc': True,
        'mocap_rotate': None,
        'fname_filter': fname_filter,
        'mocap_out_base_dir': mocap_out_base_dir,
    }

    inject_synthetic_noise_into_real_mocap(**args)


def BMLmovi():  # 120fps
    mocap_out_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'
    mocap_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/original'
    mocap_fnames = glob(osp.join(mocap_dir, 'BMLmovi', '*/*.pkl'))

    args = {
        'marker_layout_fnames': {
            'BMLmovi': '/ps/project/soma/support_files/release_soma/marker_layouts/BMLmovi/Subject_1_F_MoSh/Subject_1_F_1_smplx_finetuned.json'},
        'dataset_mocap_fnames': {'BMLmovi': mocap_fnames},
        'oc_g_bt_settings': [(5, 3, 50)],
        'mocap_unit': 'mm',
        'use_exact_num_oc': False,
        'mocap_rotate': None,
        # 'fname_filter': ['*Subject_26_F_MoSh/Subject_26_F_1*'],
        'mocap_out_base_dir': mocap_out_base_dir,
    }

    inject_synthetic_noise_into_real_mocap(**args)


def HDM05():  # 120fps

    mocap_out_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'
    mocap_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/original'
    mocap_fnames = glob(osp.join(mocap_dir, 'HDM05', '*/*.c3d'))

    args = {
        'marker_layout_fnames': {
            'HDM05': '/ps/project/soma/support_files/release_soma/marker_layouts/HDM05/dg/HDM_dg_01-04_01_120_smplx_finetuned.json'},
        'dataset_mocap_fnames': {'HDM05': mocap_fnames},
        'oc_g_bt_settings': [(5, 3, 0), (5, 0, 0), (0, 3, 0), (0, 0, 0), (5, 3, 50)],
        'mocap_unit': 'mm',
        'use_exact_num_oc': False,
        'mocap_rotate': None,
        # 'fname_filter': ['*bk/*', '*dg/*', '*mm/*', '*tr/*'],
        'fname_filter': ['*mm/HDM_mm_05-03_01_120*'],
        'mocap_out_base_dir': mocap_out_base_dir,
    }

    inject_synthetic_noise_into_real_mocap(**args)


def HDM05_attention_span():  # 120fps

    mocap_out_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'
    mocap_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/original'
    mocap_fnames = glob(osp.join(mocap_dir, 'HDM05', '*/*.c3d'))

    args = {
        'marker_layout_fnames': {
            'HDM05_attention_span': '/ps/project/soma/support_files/release_soma/marker_layouts/HDM05/dg/HDM_dg_01-04_01_120_smplx_finetuned.json'},
        'dataset_mocap_fnames': {'HDM05': mocap_fnames},
        'oc_g_bt_settings': [(0, 0, 0), ],
        'mocap_unit': 'mm',
        'use_exact_num_oc': False,
        'mocap_rotate': None,
        'fname_filter': ['*bk/*', '*dg/*', '*mm/*', '*tr/*'],
        'mocap_out_base_dir': mocap_out_base_dir,
    }

    inject_synthetic_noise_into_real_mocap(**args)


def HDM05_varied_marker_layout():
    mocap_out_base_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/with_synthetic_noise'
    mocap_dir = '/ps/project/soma/support_files/release_soma/evaluation_mocaps/original'
    mocap_fnames = glob(osp.join(mocap_dir, 'HDM05', '*/*.c3d'))

    marker_layout_fnames = {osp.basename(fname).replace('.json', ''): fname for fname in
                            glob(
                                '/ps/project/soma/support_files/release_soma/marker_layouts/HDM05/dg/modified/*.json')}

    args = {
        'marker_layout_fnames': marker_layout_fnames,
        'dataset_mocap_fnames': {'HDM05': mocap_fnames},
        'oc_g_bt_settings': [(5, 3, 0), ],
        'mocap_unit': 'mm',
        'use_exact_num_oc': False,
        'mocap_rotate': None,
        'fname_filter': ['*bk/*', '*dg/*', '*mm/*', '*tr/*'],
        'mocap_out_base_dir': mocap_out_base_dir,
    }

    inject_synthetic_noise_into_real_mocap(**args)


if __name__ == '__main__':
    # KIT()
    BMLrub()
    # BMLrub_test_ghorbani_permutation_2019()
    # BMLmovi()
    # HDM05()
    # HDM05_attention_span()
    # HDM05_varied_marker_layout()
