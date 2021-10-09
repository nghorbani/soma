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
import os.path as osp
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.labels_map import general_labels_map
from soma.data.amass_marker_noise_model import amass_marker_noise_model
from soma.data.mocap_noise_tools import make_ghost_points, occlude_markers


class MoCapSynthesizer(Dataset):
    """
    This is a dataloder supposed to be used for training SOMA.
    The dataloader expects to receive already generated markers and adds random noise to every datapoint
    """

    def __init__(self, marker_dataset_dir: Union[str, Path], marker_layout_fnames: List[Union[str, Path]] = None,
                 num_ghost_max: int = 0, num_occ_max: int = 0, num_btraj_max: int = 30,
                 marker_noise_var: int = 0, amass_marker_noise_dir: Union[str, Path] = None,
                 limit_real_data: float = 1.0, limit_synt_data: float = 1.0,
                 ghost_distribution: str = 'spherical_gaussian',
                 ):
        assert osp.exists(marker_dataset_dir), FileNotFoundError(
            f'marker_dataset_dir does not exist: {marker_dataset_dir}')
        expected_split_names = ['train', 'vald', 'test']
        assert np.any([marker_dataset_dir.endswith(k) for k in expected_split_names]), \
            ValueError(
                f'marker_dataset_dir should include one of the expected_split_names: {expected_split_names}')
        self.split_name = split_name = marker_dataset_dir.split('/')[-1]

        assert num_btraj_max == 0, ValueError(
            f'The current data loader is per-frame, '
            f'hence broken trajectories doesnt make sense; i.e. num_btraj_max = {num_btraj_max}')

        self.use_naive_marker_pos_noise = True if marker_noise_var != 0.0 else False
        self.use_model_based_marker_pos_noise = True if amass_marker_noise_dir else False

        logger.debug(
            f'Will produce augmented mocap point cloud data for split {split_name}: '
            f'num_ghost_max = {num_ghost_max}, num_occ_max = {num_occ_max}, num_btraj_max = {num_btraj_max}, '
            f'marker_noise_var = {marker_noise_var:.2e}, '
            f'use_model_based_marker_pos_noise = {self.use_model_based_marker_pos_noise}')
        ds = {}
        for data_fname in glob.glob(osp.join(marker_dataset_dir, '*.pt')):
            k = osp.basename(data_fname).replace('.pt', '')
            ds[k] = torch.load(data_fname).type(torch.float32)

        logger.debug(
            f'split_name = {split_name}, limit_real_data = {limit_real_data}, limit_synt_data = {limit_synt_data}')
        n_data = len(ds['data_is_real'])
        real_data_ids = np.arange(n_data)[ds['data_is_real'] == True]
        synth_data_ids = np.arange(n_data)[ds['data_is_real'] == False]
        if limit_real_data < 1.0 and split_name == 'train':
            np.random.seed(100)
            n_real_data_init = len(real_data_ids)
            real_data_ids = np.random.choice(real_data_ids, int(len(real_data_ids) * limit_real_data), replace=False)
            logger.debug(
                f'Chosen {limit_real_data:.2f} of real data point #{len(real_data_ids)} of #{n_real_data_init}')

        if limit_synt_data < 1.0 and split_name == 'train':
            np.random.seed(100)
            n_synth_data_init = len(synth_data_ids)
            synth_data_ids = np.random.choice(synth_data_ids, int(len(synth_data_ids) * limit_synt_data), replace=False)
            logger.debug(
                f'Chosen {limit_synt_data:.2f} of synthetic data point #{len(synth_data_ids)} of #{n_synth_data_init}')

        logger.debug('dimensions of loaded data: {}'.format({k: v.shape for k, v in ds.items()}))

        data_ids = np.concatenate([real_data_ids, synth_data_ids])
        self.ds = {k: v[data_ids] for k, v in ds.items()}

        ds_cfg_fname = glob.glob(osp.join(marker_dataset_dir, '..', '*.yaml'))
        assert len(ds_cfg_fname) > 0, FileNotFoundError(
            f'Could not find the dataset settings at {marker_dataset_dir + "/.."}')
        logger.info(f'loaded dataset settings from {ds_cfg_fname}')
        cfg_initial = OmegaConf.load(ds_cfg_fname[0])
        cfg_overload = OmegaConf.create(
            {'marker_dataset_dir': marker_dataset_dir, 'marker_layouts': marker_layout_fnames,
             'num_ghost_max': num_ghost_max, 'num_occ_max': num_occ_max, 'num_btraj_max': num_btraj_max,
             'marker_noise_var': marker_noise_var})
        self.cfg = OmegaConf.merge(cfg_initial, cfg_overload)

        logger.debug(
            f'Split {split_name}: Loaded #{self.__len__()} data points from marker_dataset_dir {marker_dataset_dir}.')
        logger.debug(
            f'Split {split_name}: #{len(real_data_ids)} ({len(real_data_ids) / self.__len__():.2f}) real '
            f'and #{len(synth_data_ids)} ({len(synth_data_ids) / self.__len__():.2f}) synthetic data points.')

        if self.cfg.use_real_data_for and split_name in self.cfg.use_real_data_for:
            logger.debug(f'real_marker_amass_splits: {self.cfg.real_marker_amass_splits[split_name]}')

        if self.cfg.use_synt_data_for and split_name in self.cfg.use_synt_data_for:
            logger.debug(f'synthetic_body_amass_splits: {self.cfg.synthetic_body_amass_splits[split_name]}')

        assert self.cfg.num_timeseq_frames == 1, NotImplementedError(
            'Check occlusion and ghost noise component for time series data')

        # self.superset_meta = MoCapSynthesizer.load_superset(self.cfg.superset_fname)#, body_parts=body_parts)
        self.superset_meta = marker_layout_load(self.cfg.superset_fname)

        self.superset_labels = list(self.superset_meta['marker_colors'].keys())

        self.num_superset_labels = len(self.superset_labels)
        self.nan_class_id = self.superset_labels.index('nan')

        # for k,v in superset_meta.items(): self.__setattr__(k,v)

        if marker_layout_fnames is None:
            self.marker_layouts = {'superset': torch.ones(len(self.superset_labels[:-1])).type(torch.bool)}
            logger.debug(f'Parameter marker_layouts was not passed so will use the superset {self.cfg.superset_fname}')
        else:
            self.marker_layouts = {}
            for marker_layout_fname in marker_layout_fnames:
                logger.debug(f'Loading marker layout: {marker_layout_fname}')
                marker_layout_name = osp.basename(marker_layout_fname).split('.')[0]
                self.marker_layouts[marker_layout_name] = []

                marker_meta = marker_layout_load(marker_layout_fname, labels_map=general_labels_map)
                cur_labels = [self.superset_labels[:-1].index(l) for l in list(marker_meta['marker_vids'].keys())]
                cur_labels_mask = torch.zeros(len(self.superset_labels[:-1]))  # .type(torch.long)
                cur_labels_mask[cur_labels] = True
                self.marker_layouts[marker_layout_name] = cur_labels_mask

        # if not (self.use_naive_marker_pos_noise and self.use_model_based_marker_pos_noise):
        #     logger.debug('Both marker_noise_var and amass_marker_noise_fname are asked to be used. amass_marker_noise_fname takes precedence')

        if self.use_model_based_marker_pos_noise:
            self.model_based_marker_noise_var = amass_marker_noise_model(
                osp.join(amass_marker_noise_dir, split_name, 'amass_marker_noise_model.npz'))
        if self.use_naive_marker_pos_noise:
            self.marker_noise_var = torch.Tensor(np.array(marker_noise_var)).type(torch.float)

        self.label_body_part_mask = {k: torch.from_numpy(v) for k, v in self.superset_meta['marker_type_mask'].items()}
        self.num_ghost_max = num_ghost_max
        self.num_occ_max = num_occ_max
        self.num_btraj_max = num_btraj_max
        self.ghost_distribution = ghost_distribution
        self.enable_prop = True if 'prop_markers' in ds else False
        self.num_prop_marker_max = 0
        if self.enable_prop:
            self.num_prop_marker_max = self.ds['prop_markers'].shape[-1] // 3
            self.prop_label_ids = torch.from_numpy(
                np.array([self.nan_class_id for _ in range(self.num_prop_marker_max)])).type(torch.long)
            self.temp_prop_mask = torch.zeros_like(self.prop_label_ids)

        self.label_ids = torch.from_numpy(np.array(list(range(self.num_superset_labels - 1)))).type(torch.long)

        self.ghost_label_ids = torch.from_numpy(
            np.array([self.nan_class_id for _ in range(self.num_ghost_max * len(self.label_body_part_mask))])).type(
            torch.long)
        self.temp_ghost_mask = torch.zeros_like(self.ghost_label_ids)
        ## preparing active sampling mechanism
        self._data_error = np.ones(self.__len__()) * 1e18  # all datapoints are equally erroneous
        self.logger = logger

    def set_data_error(self, dIds, errs):
        self._data_error[dIds] = errs

    def __len__(self):
        k = 'trans'
        # k = list(self.ds.keys())[0]
        return len(self.ds[k])

    def __getitem__(self, dIdx):
        return self.fetch_data(dIdx)

    def fetch_data(self, dIdx):
        '''
        Each data point is Txdim where T is n_frames
        Parameters
        ----------
        dIdx

        Returns
        -------

        '''

        np.random.seed(None)

        data = {k: self.ds[k][dIdx] for k in self.ds.keys()}
        # if the picked data point is real then should not pick a markerset
        # for synthetic mocap all markers are generated and dropped during loading
        # for real data not all data points exist
        # using this any datapoint who already has zeros in it should be real data

        data_is_real = data['data_is_real'].type(torch.bool)

        markers = data.pop('markers').view(-1, 3)  # .clone()
        markers_untouched = markers.clone()
        if data_is_real:  # not all superset markers are available in real data
            # self.logger.debug('This batch uses real data')
            markers_available = ((markers == 0.0).sum(-1) == 3)
            markers_not_available = torch.logical_not(markers_available)
        else:
            # This means the data is synthetic. Since we place all superset markers on the body so all of them should be non-zero
            marker_layout_name = np.random.choice(list(self.marker_layouts.keys()))
            # self.logger.debug('Chosen {} for this batch'.format(marker_layout_name))
            markers_not_available = torch.logical_not(self.marker_layouts[marker_layout_name])
            markers_available = torch.logical_not(markers_not_available)
            markers[markers_not_available] = 0.0

        ## add 3d noise to marker data
        if not data_is_real:
            if self.use_model_based_marker_pos_noise:
                markers[markers_available] += self.model_based_marker_noise_var()[0, markers_available]
            if self.use_naive_marker_pos_noise:
                markers[markers_available] += \
                    markers.new(np.abs(np.random.normal(size=markers.shape)).astype(np.float))[
                        markers_available] * self.marker_noise_var

        labels_not_available = markers_not_available.clone().type(torch.float)

        label_ids = self.label_ids.clone()

        if self.num_occ_max != 0:  # this happens first to have part specific occlusion
            for body_part, marker_mask in self.label_body_part_mask.items():
                part_specific_markers = occlude_markers(markers[marker_mask],
                                                        num_occ=np.random.choice(self.num_occ_max + 1))
                markers[marker_mask] = part_specific_markers

            cur_zero_mask = ((markers == 0.0).sum(-1) == 3)  # .type(torch.bool)
            labels_not_available[cur_zero_mask] = True

        if self.num_ghost_max != 0:
            ghost_markers = []
            for body_part, marker_mask in self.label_body_part_mask.items():
                selected_markers = markers[marker_mask]
                zero_mask = torch.logical_not((selected_markers == 0.0).sum(-1) == 3)  # .type(torch.bool)

                part_specific_ghost_markers = make_ghost_points(selected_markers[zero_mask][None],
                                                                num_ghost_max=np.random.choice(self.num_ghost_max + 1),
                                                                ghost_distribution=self.ghost_distribution,
                                                                use_upto_num_ghost=True)
                if part_specific_ghost_markers is not None:
                    ghost_markers.append(part_specific_ghost_markers[0])

            if len(ghost_markers) > 0:
                ghost_markers = torch.cat(ghost_markers, dim=0)

                num_skipped_ghosts = self.ghost_label_ids.shape[0] - ghost_markers.shape[0]
                ghost_markers = torch.cat([ghost_markers,
                                           ghost_markers.new(np.zeros([num_skipped_ghosts, 3]))
                                           ], dim=0)
            else:
                ghost_markers = markers.new(np.zeros([self.ghost_label_ids.shape[0], 3]))

            markers = torch.cat([markers, ghost_markers], dim=0)
            label_ids = torch.cat([label_ids, self.ghost_label_ids], dim=-1)  # .type(torch.long)

        if self.enable_prop:
            prop_markers = data['prop_markers'].view(self.num_prop_marker_max, 3)
            prop_markers = occlude_markers(prop_markers,
                                           num_occ=np.random.choice(self.num_occ_max + 1))
            markers = torch.cat([markers, prop_markers], dim=0)
            label_ids = torch.cat([label_ids, self.prop_label_ids], dim=-1)  # .type(torch.long)

        zero_mask = ((markers == 0.0).sum(-1) == 3)  # .type(torch.bool)
        label_ids[zero_mask] = self.nan_class_id

        permvecs = np.random.permutation(markers.shape[0])

        markers = markers[permvecs]
        label_ids = label_ids[permvecs]
        label_one_hots = torch.nn.functional.one_hot(label_ids, self.num_superset_labels).type(torch.float)

        # if self.num_btraj_max != 0 and T>1: this wont be used for per-frame training
        #     markers, label_ids = break_trajectories(markers, label_ids, nan_class_id=self.nan_class_id, num_btraj_max=self.num_btraj_max)
        #     zero_mask = ((markers == 0.0).sum(-1) == 3)  # .type(torch.bool)

        nan_mask = label_ids == self.nan_class_id
        label_weights = torch.ones_like(label_ids, dtype=torch.float)
        # label_nan_wt =  torch.div(1., torch.from_numpy(np.where(nan_mask.sum(-1) == 0, 1, nan_mask.sum(-1)))) # if num nan is zero make it one
        # label_nan_wt[torch.isnan(label_nan_wt)] = 1.0
        # label_nan_wt = nan_mask.sum(-1).type(torch.float)

        label_nan_wt = torch.div(1., nan_mask.sum(-1)) if nan_mask.sum(-1) != 0 else label_weights.new([1.0])
        # for t in torch.arange(T):
        label_weights[label_ids == self.nan_class_id] = label_nan_wt

        label_dustbin = torch.cat([labels_not_available,
                                   (self.num_superset_labels -
                                    labels_not_available.sum(-1).type(torch.float)).view(
                                       -1)],
                                  dim=-1)
        # (markers_available.shape[-1] - markers_not_available.sum(-1).type(torch.float)).view(-1,1)], dim=-1)#  ICCV'21

        # a = torch.cat([label_one_hots, label_dustbin.view(T, 1, -1)], dim=1)[0]
        # for i in range(a.shape[0]-1):
        #     assert a[i].sum() == 1, ValueError('row {} sums to {}'.format(i, a[i].sum()))
        # for i in range(a.shape[1]-1):
        #     assert a[:,i].sum() == 1, ValueError('column {} sums to {}'.format(i, a[:,i].sum()))
        #
        # assert a[-1].sum() == a.shape[1]
        # assert a[:,-1].sum() == a.shape[0]

        new_data = {}
        new_data['points'] = markers.view(-1, 3)  # permuted markers # Todo: how come points dont have seq_len?

        # new_data['nozero_mask'] = ~zero_mask

        # new_data['zero_mask'] = zero_mask
        new_data['nan_mask'] = nan_mask

        new_data['aug_asmat'] = torch.cat([label_one_hots, label_dustbin.view(1, -1)], dim=0)

        label_weights = label_weights.unsqueeze(-1)
        new_data['aug_asmat_weights'] = torch.cat([label_weights.expand(-1, label_dustbin.shape[-1]),
                                                   torch.ones_like(label_dustbin).view(1, -1) * label_nan_wt], dim=0)

        new_data['label_ids'] = label_ids

        new_data['data_idx'] = torch.from_numpy(np.array(dIdx)).view(1)

        new_data['markers_orig'] = markers_untouched

        for body_part, marker_mask in self.label_body_part_mask.items():
            if self.enable_prop:
                cur_marker_mask = torch.cat([marker_mask, self.temp_ghost_mask, self.temp_prop_mask], dim=0)[permvecs]
            else:
                cur_marker_mask = torch.cat([marker_mask, self.temp_ghost_mask], dim=0)[permvecs]

            cur_marker_mask[nan_mask] = 0
            new_data[f'{body_part}_mask'] = cur_marker_mask.type(torch.float)

        data.update(new_data)
        return data
