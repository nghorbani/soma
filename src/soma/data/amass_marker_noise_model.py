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

import glob
import os
from os import path as osp
from pathlib import Path
from typing import Union, Dict

import numpy as np
import torch
from human_body_prior.tools.omni_tools import create_list_chunks
from human_body_prior.tools.omni_tools import makepath
from loguru import logger

from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.labels_map import general_labels_map
from moshpp.mosh_head import MoSh
from soma.data.marker_dataset import sort_markers_like_superset


def amass_marker_noise_model_exists(amass_marker_noise_dir: Union[str, Path], amass_splits: Dict[str, list]) -> bool:
    """
    Determine whether AMASS noise model exists for the given amass splits.
    If a split name is empty it is assumed not wanted.
    Args:
        amass_marker_noise_dir:
        amass_splits:

    Returns:
        True/False
    """
    done = []
    for split_name in amass_splits:
        if amass_splits[split_name] is None: done.append(True)
        amass_marker_noise_fname = osp.join(amass_marker_noise_dir, split_name, 'amass_marker_noise_model.npz')
        if osp.exists(amass_marker_noise_fname):
            done.append(True)
        else:
            done.append(False)

    if np.all(done):
        logger.debug(f'AMASS noise model already exists at {amass_marker_noise_fname}')
        return True

    return False


def prepare_amass_marker_noise_model(amass_marker_noise_dir: Union[str, Path],
                                     superset_fname: Union[str, Path],
                                     amass_splits: Dict[str, list],
                                     amass_dir: Union[str, Path],
                                     num_timeseq_frames: int = 1,
                                     unified_frame_rate: int = 30,
                                     num_frames_overlap: int = 0,
                                     babel: Dict[str, list] = None):
    """
    We copy the noise for each label from the real AMASS mocap markers to help generalize
    to mocap hardware differences. We create a database of the differences between the simulated and actual markers of
    AMASS and draw random samples from this noise model to add to the synthetic marker positions.

    Implementation idea is as follows: for a real mocap sequence, we have a corresponding mosh fit which has pairs of
    simulated and observed (real) markers.  We can use this in two ways:
    First fit a distribution to the differences in the markers and then use this to sample from to add noise.
    Another simpler way implemented here assumes that the observed errors represent the distribution and we literally draw samples from that.
    Specifically, take a random frame of mocap.  Take the vector displacements between the simulated and real markers.
    Now, for the fame of synthetic marker we created for soma, to which we want to add noise,
    we add vector of displacements corresponding to labels of the synthetic markers.
    algo:
        # 1) load marker layout and create a marker noise dictionary;
             i.e. <labels:[list of noise sequences with length num_timeseq_frames]>
        # 2) load mosh pkl files with simulated and real marker data.
             You might need to create these first by running mosh on actual mocap markers of amass subsets
        # 3) for each mosh fname get observed and simulated markers and replace simulated markers in place of real ones
             whenever real markers not available; i.e. occluded
        # 4) compute the distance and save a windowed copy in the marker noise dictionary

    Args:
        amass_marker_noise_dir: the output directory for amass noise model
        superset_fname: the path to superset marker layout
        amass_splits: {'vald':[], 'train':[]}
        amass_dir:  amass directory where released npz files and original mosh pkl files exist.
                    for mosh pkl files one might need to run mosh on real markers of the original dataset.
        num_timeseq_frames: if need to train a sequence model this should be larger than 1
        unified_frame_rate:
        num_frames_overlap:
        babel: whether to use babel[2]. it should be a dictionary with key being the split name and
               values list of npz files from amass

    References:
        [1] https://amass.is.tue.mpg.de/
        [2] https://babel.is.tue.mpg.de/
    """
    assert superset_fname.endswith('.json')
    if amass_marker_noise_model_exists(amass_marker_noise_dir, amass_splits): return

    marker_meta = marker_layout_load(superset_fname, labels_map=general_labels_map)
    superset_labels = list(marker_meta['marker_vids'].keys())

    log_fname = makepath(amass_marker_noise_dir, 'dataset.log', isfile=True)
    log_format = "{module}:{function}:{line} -- {level} -- {message}"
    ds_logger_id = logger.add(log_fname, format=log_format, enqueue=True)

    logger.debug(
        f'Creating amass marker noise model for superset {superset_fname} which has {len(superset_labels)} markers')
    marker_noise_map = {l: [] for l in superset_labels}
    for split_name in amass_splits.keys():

        if amass_splits[split_name] is None: continue
        amass_marker_noise_fname = makepath(amass_marker_noise_dir, split_name, 'amass_marker_noise_model.npz',
                                            isfile=True)
        for ds_name in amass_splits[split_name]:
            mosh_stageii_pkl_fnames, used_babel = [], False
            if babel and ds_name in babel:
                mosh_stageii_pkl_fnames = [fname.replace('.npz', '.pkl') for fname in babel[ds_name]]
                if mosh_stageii_pkl_fnames: used_babel = True

            if not mosh_stageii_pkl_fnames:
                subset_dir = os.path.join(amass_dir, ds_name)
                mosh_stageii_pkl_fnames = glob.glob(os.path.join(subset_dir, '*/*_stageii.pkl'))

            if len(mosh_stageii_pkl_fnames) == 0:
                logger.error(f'No mosh_stageii result found for {ds_name} at {subset_dir}')
                continue

            mosh_stageii_pkl_fnames = np.random.choice(mosh_stageii_pkl_fnames,
                                                       min([20, len(mosh_stageii_pkl_fnames)]), replace=False).tolist()

            logger.debug(
                f'Found #{len(mosh_stageii_pkl_fnames)} for split {split_name} from ds_name {ds_name}. used_babel={used_babel}')
            for pkl_fname in mosh_stageii_pkl_fnames:

                mosh_data = MoSh.load_as_amass_npz(stageii_pkl_data_or_fname=pkl_fname, include_markers=True)
                ds_rate = max(1, int(mosh_data['mocap_frame_rate'] // unified_frame_rate))

                markers_sim = sort_markers_like_superset(mosh_data['markers_sim'][::ds_rate],
                                                         mosh_data['labels_obs'][::ds_rate],
                                                         superset_labels=superset_labels)
                markers_obs = sort_markers_like_superset(mosh_data['markers_obs'][::ds_rate],
                                                         mosh_data['labels_obs'][::ds_rate],
                                                         superset_labels=superset_labels)

                for tIds in create_list_chunks(range(len(markers_sim)),
                                               group_size=num_timeseq_frames,
                                               overlap_size=num_frames_overlap,
                                               cut_smaller_batches=True):
                    for lId, l in enumerate(superset_labels):
                        if l == 'nan': continue
                        if np.all(markers_obs[tIds, lId] == 0): continue
                        marker_noise_map[l].append(
                            (markers_obs[tIds, lId] - markers_sim[tIds, lId]).astype(np.float))

        uncovered_labels = [k for k, v in marker_noise_map.items() if len(v) == 0]
        if len(uncovered_labels):
            logger.error(f'split_name {split_name}: No sequence found for labels {uncovered_labels}')

        # make all equal by oversampling
        # print({k: len(v) for k, v in marker_noise_map.items()})
        num_max_seq_per_label = max([len(v) for k, v in marker_noise_map.items()])
        for k, v in marker_noise_map.items():
            if len(v) < num_max_seq_per_label:
                if len(v) == 0:
                    marker_noise_map[k] = [np.zeros([num_timeseq_frames, 3]) for _ in range(num_max_seq_per_label)]
                else:
                    over_sampled_ids = np.random.choice(range(len(v)), num_max_seq_per_label - len(v))
                    for i in over_sampled_ids:
                        marker_noise_map[k].append(marker_noise_map[k][i])

        amass_marker_noise_model = np.zeros([num_max_seq_per_label, len(superset_labels), num_timeseq_frames, 3],
                                            dtype=np.float)
        for i, l in enumerate(superset_labels):
            np.random.shuffle(marker_noise_map[l])
            amass_marker_noise_model[:, i] = np.stack(marker_noise_map[l])

        amass_marker_noise_model = amass_marker_noise_model.transpose([0, 2, 1, 3])

        # print({k: len(v) for k, v in marker_noise_map.items()})

        np.savez(amass_marker_noise_fname, amass_marker_noise_model=amass_marker_noise_model)
        logger.debug(f'Created AMASS marker noise model at: {amass_marker_noise_fname}')
    logger.remove(ds_logger_id)


def amass_marker_noise_model(amass_marker_noise_fname: Union[str, Path]):
    assert osp.exists(amass_marker_noise_fname), FileNotFoundError(f'Could not find {amass_marker_noise_fname}')

    label_noise_map = torch.from_numpy(np.load(amass_marker_noise_fname)['amass_marker_noise_model']).type(torch.float)

    def produce_once():
        i = np.random.choice(len(label_noise_map))
        return label_noise_map[i]

    return produce_once
