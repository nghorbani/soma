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
from pathlib import Path
from typing import Union, Dict

import numpy as np
import tables as pytables
import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import makepath
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from soma.data.body_synthesizer import FullBodySynthesizer, betas_populate_source
from soma.data.body_synthesizer import body_populate_source, face_populate_source, hand_populate_source
from soma.data.marker_dataset import dataset_exists


def prepare_synthetic_body_dataset(body_dataset_dir: Union[str, Path],
                                   amass_splits: Dict[str, list],
                                   amass_npz_dir: Dict[str, list],
                                   surface_model_fname: Union[str, Path],
                                   unified_frame_rate: int = 30,
                                   num_hand_var_perseq: int = 15,
                                   num_betas: int = 10,
                                   num_expressions: int = 80,
                                   rnd_zrot: bool = True,
                                   gender: str = 'neutral',
                                   animate_face: bool = True,
                                   animate_hand: bool = True,
                                   num_timeseq_frames: int = 1,
                                   num_frames_overlap: int = 0,
                                   babel: Dict[str, list] = None,
                                   augment_by_temporal_inversion: bool = False):
    """
    To learn a robust model, we exploit AMASS [1] in neutral gender SMPL-X body model
    and sub-sample to a unified 30 Hz. To be robust to subject body shape we generate AMASS motions for 3664 bodies
    from the CAESAR dataset [2]. Specifically, for training we take parameters from the following mocap sub-datasets
    of AMASS: CMU [9], Transitions [23] and Pose Prior [5].
    For validation we use HumanEva [40], ACCAD [4], and
    TotalCapture [25].

    References:
    [1] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black.
        AMASS: Archive of motion capture as surface shapes. In 2019 IEEE/CVF International
        Conference on Computer Vision (ICCV), pages 5441–5450, Oct. 2019.

    [2] K. Robinette, S. Blackwell, H. Daanen, M. Boehmer, S. Fleming, T. Brill, D. Hoeferlin, and D. Burnsides. Civilian
        American and European Surface Anthropometry Resource (CAESAR) final report. Technical Report AFRL-HE-WPTR-
        2002-0169, US Air Force Research Laboratory, 2002.

    Args:
        body_dataset_dir:
        amass_splits:
        amass_npz_dir:
        surface_model_fname:
        unified_frame_rate:
        num_hand_var_perseq:
        num_betas:
        num_expressions:
        rnd_zrot:
        gender:
        animate_face:
        animate_hand:
        num_timeseq_frames:
        num_frames_overlap:
        babel:
        augment_by_temporal_inversion:

    Returns:

    """
    if dataset_exists(body_dataset_dir, split_names=['train', 'vald']):
        # we do not produce synthetic tests data. we tests only on real data
        logger.debug(f'Synthetic body dataset already exists at {body_dataset_dir}')
        return

    if np.all([(amass_splits[split_name] is not None) and (len(amass_splits[split_name]) != 0) for split_name in
               ['vald', 'train']]):
        assert len(set(amass_splits['train'] + amass_splits['vald'])) == len(set(amass_splits['train'])) + len(
            set(amass_splits['vald'])), \
            ValueError('Validation and training sets have overlapping elements')

    log_fname = makepath(body_dataset_dir, 'dataset.log', isfile=True)
    log_format = "{module}:{function}:{line} -- {message}"
    ds_logger_id = logger.add(log_fname, format=log_format, enqueue=True)

    surface_model_type = BodyModel(surface_model_fname).model_type
    acceptenum_total_attention_feat_types = ['smplx', 'animal_dog']
    assert np.any([k in surface_model_type for k in acceptenum_total_attention_feat_types]), ValueError(
        f'model_type should be one of {acceptenum_total_attention_feat_types}')

    logger.debug(f'Dumping SOMA synthetic_body_dataset at {body_dataset_dir}')
    logger.debug(
        f'These parameters will be used: unified_frame_rate={unified_frame_rate}, '
        f'num_hand_var_perseq={num_hand_var_perseq}, animate_face={animate_face}, '
        f'animate_hand={animate_hand},num_timeseq_frames={num_timeseq_frames}, num_frames_overlap={num_frames_overlap}')

    class BodyDataSetRow(pytables.IsDescription):
        betas = pytables.Float32Col(num_timeseq_frames * num_betas)  # float  (single-precision)
        trans = pytables.Float32Col(num_timeseq_frames * 3)  # float  (single-precision)
        root_orient = pytables.Float32Col(num_timeseq_frames * 3)  # float  (single-precision)
        if surface_model_type == 'animal_dog':
            raise NotImplementedError('This functionality is not released for current SOMA.')

        elif surface_model_type == 'smplx':
            pose_body = pytables.Float32Col(num_timeseq_frames * 21 * 3)  # float  (single-precision)
            if animate_hand:
                raise NotImplementedError('This functionality is not released for current SOMA.')
            if animate_face:
                raise NotImplementedError('This functionality is not released for current SOMA.')

            # joints = pytables.Float32Col(num_timeseq_frames * 165)  # float  (single-precision)

    mocap_ds = FullBodySynthesizer(surface_model_fname,
                                   unified_frame_rate=unified_frame_rate,
                                   num_hand_var_perseq=num_hand_var_perseq,
                                   num_betas=num_betas,
                                   num_expressions=num_expressions,
                                   augment_by_temporal_inversion=augment_by_temporal_inversion)

    body_npz_fnames = body_populate_source(amass_npz_dir, amass_splits, babel=babel)
    face_npz_fnames = face_populate_source() if animate_face else {k: None for k in body_npz_fnames.keys()}
    hand_frames = hand_populate_source() if animate_hand else {k: None for k in body_npz_fnames.keys()}
    betas = betas_populate_source(amass_npz_dir, amass_splits, gender, num_betas)

    for split_name in ['train', 'vald']:  # for testing we use only real data
        logger.debug(f'--------------- Dataset Split {split_name.upper()} ------------------')
        ds_iter = mocap_ds.sample_mocap_windowed(body_fnames=body_npz_fnames[split_name],
                                                 face_fnames=face_npz_fnames[split_name],
                                                 hand_frames=hand_frames[split_name],
                                                 betas=betas[split_name],
                                                 num_timeseq_frames=num_timeseq_frames,
                                                 num_frames_overlap=num_frames_overlap,
                                                 rnd_zrot=rnd_zrot)
        h5_fpath = makepath(body_dataset_dir, split_name, 'data.h5', isfile=True)
        if not os.path.exists(h5_fpath):
            with pytables.open_file(h5_fpath, mode="w") as h5file:
                table = h5file.create_table('/', 'data', BodyDataSetRow)
                for data in tqdm(ds_iter):
                    for k in BodyDataSetRow.columns.keys():
                        table.row[k] = c2c(data[k]).reshape(-1)
                    table.row.append()
                table.flush()

        assert os.path.exists(h5_fpath), ValueError(f'Data file {h5_fpath} does not exist!')
        with pytables.open_file(h5_fpath, mode="r") as h5file:
            data = h5file.get_node('/data')
            logger.debug(f'Dumping {len(data):d} data points for {split_name} split as final pytorch pt files.')

            data_dict = {k: [] for k in data.colnames}
            for id in range(len(data)):
                cdata = data[id]
                for k in data_dict.keys():
                    data_dict[k].append(cdata[k])

        for k, v in data_dict.items():
            outfname = makepath(body_dataset_dir, split_name, f'{k}.pt', isfile=True)
            if os.path.exists(outfname): continue
            torch.save(torch.from_numpy(np.asarray(v)), outfname)

    body_dataset_cfg = OmegaConf.create({
        'amass_splits': OmegaConf.to_object(amass_splits),
        'gender': gender,
        'unified_frame_rate': unified_frame_rate,
        'num_hand_var_perseq': num_hand_var_perseq,
        'num_betas': num_betas,
        'num_expressions': num_expressions,
        'surface_model_type': surface_model_type,
        'num_timeseq_frames': num_timeseq_frames,
        'num_frames_overlap': num_frames_overlap,
        'animate_face': animate_face,
        'animate_hand': animate_hand,
    })
    OmegaConf.save(body_dataset_cfg, f=makepath(body_dataset_dir, 'settings.yaml', isfile=True))

    logger.debug(f'body_dataset_dir: {body_dataset_dir}')
    logger.remove(ds_logger_id)
