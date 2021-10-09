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
from collections import OrderedDict
from os import path as osp
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import pytorch_lightning as pl
import torch
from human_body_prior.tools.omni_tools import flatten_list
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from moshpp.chmosh import mosh_stagei
from moshpp.marker_layout.edit_tools import SuperSet
from moshpp.marker_layout.edit_tools import marker_layout_as_mesh, marker_layout_load
from moshpp.marker_layout.edit_tools import marker_layout_write
from moshpp.marker_layout.edit_tools import merge_marker_layouts
from moshpp.marker_layout.labels_map import general_labels_map
from moshpp.mosh_head import MoSh
from moshpp.tools.mocap_interface import MocapSession
from soma.data.amass_marker_noise_model import prepare_amass_marker_noise_model
from soma.data.marker_dataset import prepare_marker_dataset
from soma.data.mocap_dataset import MoCapSynthesizer
from soma.data.synthetic_body_dataset import prepare_synthetic_body_dataset


def prepare_training_superset(marker_layout_fnames: List[Union[str, Path]],
                              superset_fname: Union[str, Path]) -> SuperSet:
    if osp.exists(superset_fname):
        assert os.path.exists(superset_fname), FileNotFoundError(f'Superset could not be found {superset_fname}')
        superset_meta = marker_layout_load(superset_fname, labels_map=general_labels_map)
        logger.debug(f'Loading superset from: {superset_fname}')

    else:
        # todo: check if all provided layouts are json files.

        marker_layout_fnames = flatten_list([glob.glob(l) for l in marker_layout_fnames])
        superset_meta = merge_marker_layouts(marker_layout_fnames, labels_map=general_labels_map)

        marker_layout_write(superset_meta, superset_fname)
        logger.debug(f'Created superset at: {superset_fname}')
        superset_meta = marker_layout_load(superset_fname, labels_map=general_labels_map)

    return superset_meta


class SOMADATAModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        logger.debug('Setting up SOMA data loader')

        seq_len = cfg.data_parms.num_timeseq_frames
        assert seq_len == 1, NotImplementedError(
            'Current SOMA is a per-frame model. num_timeseq_frames ({}) should be 1.'.format(
                cfg.data_parms.num_timeseq_frames))

        self.soma_data_id = cfg.soma.data_id
        self.marker_layout_fnames = cfg.data_parms.mocap_dataset.marker_layout_fnames

        for mid, marker_layout_fname in enumerate(self.marker_layout_fnames):
            if marker_layout_fname.endswith('.c3d'):
                self.marker_layout_fnames[mid] = self.marker_layout_from_c3d_with_mosh(marker_layout_fname)

        self.superset_meta = superset_meta = prepare_training_superset(
            marker_layout_fnames=self.marker_layout_fnames,
            superset_fname=cfg.data_parms.marker_dataset.superset_fname)

        self.superset_fname = superset_meta['marker_layout_fname']
        superset_ply_fname = self.superset_fname.replace('.json', '.ply')
        if not osp.exists(superset_ply_fname):
            res = marker_layout_as_mesh(cfg.surface_model.fname)(self.superset_fname)
            res['body_marker_mesh'].export(superset_ply_fname)
            logger.debug(f'Created {superset_ply_fname}')

        self.body_parts = cfg.model_parms.labeler.body_parts = list(superset_meta['marker_type_mask'].keys())
        self.batch_size = self.cfg.train_parms.batch_size

        self.enable_props = cfg.data_parms.marker_dataset.props.enable
        self.num_prop_marker_max = 0
        self.static_props = None
        if self.enable_props:
            raise NotImplementedError('This functionality is not released for current SOMA.')

        self.num_points = cfg.data_parms.mocap_dataset.num_ghost_max * len(self.body_parts) + \
                          self.num_prop_marker_max + \
                          len(superset_meta['marker_vids'])

        logger.debug(
            f'num_points ({self.num_points}) = num_ghost_max ({cfg.data_parms.mocap_dataset.num_ghost_max}) * '
            f'num_body_parts ({len(self.body_parts)}) + '
            f'num_prop_marker_max ({self.num_prop_marker_max}) + num_superset_labels ({len(superset_meta["marker_vids"])}) ')

        self.superset_labels = np.array(list(superset_meta['marker_colors'].keys()))
        self.num_labels = OrderedDict({k: np.sum(v) for k, v in superset_meta['marker_type_mask'].items()})

        self.example_input_array = {'points': torch.ones(self.batch_size, self.num_points, 3)}

    @rank_zero_only
    def prepare_data(self):

        cfg = self.cfg

        if cfg.data_parms.marker_dataset.use_real_data_from:
            real_marker_amass_splits = {split_name:
                                            [ds_name for ds_name in ds_names if
                                             ds_name in cfg.data_parms.marker_dataset.use_real_data_from]
                                        for split_name, ds_names in cfg.data_parms.amass_splits.items()}
        else:
            real_marker_amass_splits = cfg.data_parms.amass_splits

        prepare_synthetic_body_dataset(body_dataset_dir=cfg.dirs.body_dataset_dir,
                                       amass_splits=cfg.data_parms.amass_splits,
                                       amass_npz_dir=cfg.dirs.amass_dir,
                                       surface_model_fname=cfg.surface_model.fname,
                                       unified_frame_rate=cfg.data_parms.unified_frame_rate,
                                       num_betas=cfg.surface_model.num_betas,
                                       num_expressions=cfg.surface_model.num_expressions,
                                       gender=cfg.surface_model.gender,
                                       num_timeseq_frames=cfg.data_parms.num_timeseq_frames,
                                       num_frames_overlap=cfg.data_parms.num_frames_overlap,
                                       num_hand_var_perseq=cfg.data_parms.body_dataset.num_hand_var_perseq,
                                       rnd_zrot=cfg.data_parms.body_dataset.rnd_zrot,
                                       animate_face=cfg.data_parms.body_dataset.animate_face,
                                       animate_hand=cfg.data_parms.body_dataset.animate_hand,
                                       augment_by_temporal_inversion=cfg.data_parms.body_dataset.augment_by_temporal_inversion,
                                       )

        prepare_marker_dataset(marker_dataset_dir=cfg.dirs.marker_dataset_dir,
                               body_dataset_dir=cfg.dirs.body_dataset_dir,
                               real_marker_amass_splits=real_marker_amass_splits,
                               amass_pkl_dir=cfg.dirs.amass_dir,
                               superset_fname=self.superset_fname,
                               wrist_markers_on_stick=cfg.data_parms.marker_dataset.wrist_markers_on_stick,
                               use_real_data_for=cfg.data_parms.marker_dataset.use_real_data_for,
                               use_synt_data_for=cfg.data_parms.marker_dataset.use_synt_data_for,
                               num_random_vid_ring=cfg.data_parms.marker_dataset.num_random_vid_ring,
                               enable_rnd_vid_on_face_hands=cfg.data_parms.marker_dataset.enable_rnd_vid_on_face_hands,
                               static_props_array=self.static_props,
                               num_marker_layout_augmentation=cfg.data_parms.marker_dataset.num_marker_layout_augmentation,
                               surface_model_fname=cfg.surface_model.fname,
                               )

        if cfg.data_parms.mocap_dataset.amass_marker_noise_model.enable:
            prepare_amass_marker_noise_model(amass_marker_noise_dir=cfg.dirs.amass_marker_noise_dir,
                                             amass_splits=cfg.data_parms.mocap_dataset.amass_marker_noise_model.amass_splits,
                                             amass_dir=cfg.dirs.amass_dir,
                                             unified_frame_rate=cfg.data_parms.unified_frame_rate,
                                             superset_fname=self.superset_fname,
                                             num_timeseq_frames=cfg.data_parms.num_timeseq_frames,
                                             num_frames_overlap=cfg.data_parms.num_frames_overlap,
                                             )

    def setup(self, stage: Optional[str] = None):
        # # self.dims is returned when you call dm.size()
        # # Setting default dims here because we know them.
        # # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            for split_name in ['vald', 'train']:
                amass_marker_noise_dir = osp.join(self.cfg.dirs.amass_marker_noise_dir) if \
                    self.cfg.data_parms.mocap_dataset.amass_marker_noise_model.enable else None

                dataset = MoCapSynthesizer(
                    marker_dataset_dir=osp.join(self.cfg.dirs.marker_dataset_dir, split_name),
                    amass_marker_noise_dir=amass_marker_noise_dir,
                    limit_real_data=self.cfg.data_parms.mocap_dataset.limit_real_data,
                    limit_synt_data=self.cfg.data_parms.mocap_dataset.limit_synt_data,
                    marker_layout_fnames=self.marker_layout_fnames,
                    num_ghost_max=self.cfg.data_parms.mocap_dataset.num_ghost_max,
                    num_occ_max=self.cfg.data_parms.mocap_dataset.num_occ_max,
                    num_btraj_max=0,
                    marker_noise_var=self.cfg.data_parms.mocap_dataset.marker_noise_var,
                    ghost_distribution=self.cfg.data_parms.mocap_dataset.ghost_distribution,
                )
                assert len(dataset) != 0, ValueError('No data point available!')
                self.__setattr__(f'soma_{split_name}', dataset)
                # self.__setattr__('size'.format(split_name), dataset.shape)

    def train_dataloader(self):
        return DataLoader(self.soma_train,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=True,
                          num_workers=self.cfg.train_parms.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.soma_vald,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=False,
                          num_workers=self.cfg.train_parms.num_workers,
                          pin_memory=True)

    def marker_layout_from_c3d_with_mosh(self, mocap_fname: Union[str, Path]) -> str:
        # todo make the cfg be a moshpp section of the main soma config file
        # todo load the mocap and pass frames directly for stagei
        assert mocap_fname.endswith('.c3d')
        assert osp.exists(mocap_fname), FileNotFoundError(mocap_fname)

        surface_model_type = self.cfg.surface_model.type

        logger.info('A c3d file is given as marker layout. Running MoSh stage-i to obtain marker layout.')
        marker_layout_outfname = mocap_fname.replace('.c3d', f'_{surface_model_type}_finetuned.json')
        if osp.exists(marker_layout_outfname):
            logger.info(f'MoSh computed marker layout already exists: {marker_layout_outfname}.')
            return marker_layout_outfname

        mosh_work_base_dir = osp.join(osp.dirname(mocap_fname), f'moshpp_{surface_model_type}')
        mocap_basename = '.'.join(osp.basename(mocap_fname).split('.')[:-1])
        mosh_job = {
            'mocap.fname': mocap_fname,
            'dirs.work_base_dir': mosh_work_base_dir,
            'dirs.marker_layout_fname': osp.join(mosh_work_base_dir, f'{mocap_basename}.json'),
            'dirs.stagei_fname': osp.join(mosh_work_base_dir, f'{mocap_basename}_stagei.pkl'),
            'dirs.stageii_fname': None,
            'dirs.log_fname': osp.join(mosh_work_base_dir, f'{mocap_basename}.log'),
            'moshpp.stagei_frame_picker.type': 'manual'
        }

        num_frames = len(MocapSession(mocap_fname, mocap_unit='mm'))  # unit of mocap doesnt play a role here
        frame_ids = np.random.choice(num_frames, size=12, replace=num_frames <= 12)

        mosh_job['moshpp.stagei_frame_picker.stagei_mocap_fnames'] = [f'{mocap_fname}_{i:03d}' for i in frame_ids]

        mp = MoSh(dict_cfg=OmegaConf.to_object(self.cfg.moshpp_cfg_override), **mosh_job)

        mp.mosh_stagei(mosh_stagei)

        marker_meta = MoSh.extract_marker_layout_from_mosh(mp.stagei_fname)
        marker_layout_write(marker_meta, marker_layout_fname=marker_layout_outfname)

        marker_layout_as_mesh(self.cfg.surface_model.fname, preserve_vertex_order=True,
                              )(marker_layout_outfname,
                                marker_layout_outfname.replace('.json', '.ply'))

        return marker_layout_outfname


def create_expr_message(cfg):
    expr_msg = '------------------------------------------\n'
    expr_msg += f'[{cfg.soma.expr_id}] batch_size = {cfg.train_parms.batch_size}.\n'
    expr_msg += 'Given a 3D MoCap Point Cloud (MPC) SOMA outputs labels assigned to points.\n'
    expr_msg += 'Labeling loss is NLL.\n'
    expr_msg += f'An optimal transport layer in the output of the labeler with {cfg.model_parms.labeler.num_sinkhorn_iters} Sinkhorn iterations to encourage one label per point.\n'
    expr_msg += 'Weighting down the majority class proportionally to its over representation.\n'
    expr_msg += 'Each data point is a frame of noisy points.\n'
    expr_msg += f'Marker layout used: {cfg.data_parms.marker_dataset.superset_fname}\n'
    expr_msg += f'MoCap data parameters for split train: num_ghost_max = {cfg.data_parms.mocap_dataset.num_ghost_max}, ' \
                f'num_occ_max = {cfg.data_parms.mocap_dataset.num_occ_max},  ' \
                f'marker_noise_var = {cfg.data_parms.mocap_dataset.marker_noise_var:.2e},' \
                f' num_random_vid_ring = {cfg.data_parms.marker_dataset.num_random_vid_ring}\n'
    expr_msg += f'** Using various proportions of real ({cfg.data_parms.mocap_dataset.limit_real_data}) ' \
                f'and synthetic ({cfg.data_parms.mocap_dataset.limit_synt_data}) data for training\n'
    expr_msg += f'** Using amass_splits: {" - ".join(["{}: {}".format(k, v) for k, v in cfg.data_parms.amass_splits.items()])}\n'
    expr_msg += f'** Using gen_optimizer.args: {" - ".join(["{}: {}".format(k, v) for k, v in cfg.train_parms.gen_optimizer.args.items()])}\n'

    expr_msg += f'** Using loss weighting: {" - ".join(["{}: {:.2f}".format(k, v) for k, v in cfg.train_parms.loss_weights.items()])}\n'

    expr_msg += f'** labeler params: {" - ".join(["{}: {}".format(k, v) for k, v in cfg.model_parms.labeler.items()])}\n'

    expr_msg += f'** Props in the synthetic mocap: {" - ".join(["{}: {}".format(k, v) for k, v in cfg.data_parms.marker_dataset.procfg.items()])}'

    expr_msg += '-----------------------------------------\n'

    return expr_msg
