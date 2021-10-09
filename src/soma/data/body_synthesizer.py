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
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import torch
from human_body_prior.tools.omni_tools import create_list_chunks
from human_body_prior.tools.rotation_tools import noisy_zrot
from loguru import logger

from soma.data.sample_hand_sequences import MANO
from soma.data.sample_hand_sequences import fullrightpose2leftpose as r2l


class FullBodySynthesizer:
    def __init__(self, surface_model_fname: Union[str, Path],
                 unified_frame_rate: int = 30, num_hand_var_perseq: int = 15,
                 num_betas: int = 10, num_expressions: int = 80,
                 augment_by_temporal_inversion: bool = False,
                 wrist_markers_on_stick: bool = False):

        self.start_time = datetime.now().replace(microsecond=0)
        logger.debug('Starting to synthesize body/markers')

        self.surface_model_fname = surface_model_fname
        self.num_hand_var_perseq = num_hand_var_perseq
        self.num_betas = num_betas
        self.num_expressions = num_expressions
        self.wrist_markers_on_stick = wrist_markers_on_stick
        self.unified_frame_rate = unified_frame_rate
        self.augment_by_temporal_inversion = augment_by_temporal_inversion

        self.comp_device = torch.device("cpu")
        # usually mocap sequences are very long and they might not fit to gpu. ToDo: process in batches?
        # self.comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _body_sampler(self, npz_fnames):
        '''
        :param npz_fnames: numpy files holding poses, trans and gender
        :return:
        '''

        np.random.shuffle(npz_fnames)

        logger.debug(f'Total body parameter sequences {len(npz_fnames):05d} ')

        def get_next_seq():
            while get_next_seq.npz_id < len(npz_fnames):
                mosh_fname = npz_fnames[get_next_seq.npz_id]
                get_next_seq.npz_id += 1
                try:
                    mo = np.load(mosh_fname)
                except:
                    logger.debug(f'Problem occurred when accessing {mosh_fname}')
                    continue

                ds_rate = int(mo['mocap_frame_rate'] // self.unified_frame_rate)
                # if ds_rate == 0:
                #     logger.debug('ds_rate == 0 for {}. skipping'.format(mosh_fname))
                #     continue
                if ds_rate == 0: ds_rate = 1

                result = {
                    'pose_body': mo['pose_body'][::ds_rate],
                    'root_orient': mo['root_orient'][::ds_rate],
                    'trans': mo['trans'][::ds_rate],
                    # 'pose_body_gender': mo['gender'].tolist(),
                    # 'pose_body_fname': npz_fnames[get_next_seq.npz_id]
                }
                yield result

        get_next_seq.npz_id = 0

        return get_next_seq

    def _face_sampler(self, npz_fnames):
        
        raise NotImplementedError('This functionality is not released for current SOMA.')

    def _hand_sampler(self, handR_frames):

        raise NotImplementedError('This functionality is not released for current SOMA.')

    def _betas_sampler(self, betas, single_beta_perseq=True):

        logger.debug(f'Total beta parameters {len(betas):05d} ')

        def gen_betas(T):
            if single_beta_perseq:
                return {'betas': np.repeat(betas[np.random.choice(len(betas))][None], repeats=T, axis=0)}
            return {'betas': betas[np.random.choice(len(betas), size=T, replace=False)]}

        return gen_betas

    def sample_mocap_windowed(self, body_fnames, face_fnames=None, hand_frames=None, betas=None,
                              num_timeseq_frames=15, num_frames_overlap=8, rnd_zrot=True):

        body_sampler = self._body_sampler(body_fnames)
        face_sampler = self._face_sampler(face_fnames) if face_fnames is not None else None
        hand_sampler = self._hand_sampler(hand_frames) if hand_frames is not None else None
        betas_sampler = self._betas_sampler(betas, single_beta_perseq=True)

        # bm = BodyModel(bm_fname=self.surface_model_fname,
        #                num_betas=self.num_betas,
        #                num_expressions=self.num_expressions).to(self.comp_device)

        for id, body_parms_np in enumerate(body_sampler()):
            T = len(body_parms_np['pose_body'])
            if T <= self.num_hand_var_perseq: continue
            if face_fnames is not None: body_parms_np.update(face_sampler(T))
            if hand_frames is not None: body_parms_np.update(hand_sampler(T, self.num_hand_var_perseq))

            for tIds in create_list_chunks(range(T), num_timeseq_frames, num_frames_overlap):

                windowed_body_parms = {k: v[tIds] for k, v in body_parms_np.items() if isinstance(v, np.ndarray)}
                windowed_body_parms.update(betas_sampler(len(tIds)))
                if rnd_zrot: windowed_body_parms['root_orient'] = noisy_zrot(windowed_body_parms['root_orient'])
                # body_parms_torch = {k: torch.from_numpy(v.astype(np.float32)).to(self.comp_device)
                #                     for k, v in windowed_body_parms.items()}
                if np.any([np.any(np.isnan(v)) for v in windowed_body_parms.values()]):
                    print({k: np.any(np.isnan(v)) for k, v in windowed_body_parms.items()})
                    raise ValueError('detected nan value in marker data.')

                result = windowed_body_parms.copy()

                # result.update({'body_parms_torch': body_parms_torch,
                #                'joints': bm(betas=body_parms_torch['betas']).Jtr})

                yield result

                if self.augment_by_temporal_inversion:
                    result = {k: v[::-1] for k, v in windowed_body_parms.copy().items()}
                    # body_parms_rev_torch = {k: v.flip(0) for k, v in body_parms_torch.items()}

                    # result.update({'body_parms_torch': body_parms_rev_torch,
                    #                'joints': bm(betas=body_parms_rev_torch['betas']).Jtr
                    #                })

                    yield result  # this yields the augmentation. it is different than the one before


def body_populate_source(amass_npz_dir, amass_splits, babel=None):
    npz_fnames = {k: [] for k in amass_splits.keys()}

    for split_name in amass_splits.keys():
        if amass_splits[split_name] is None: continue
        for ds_name in amass_splits[split_name]:
            ds_npz_contents, used_babel = [], False
            if babel and ds_name in babel:
                ds_npz_contents = babel.get(ds_name, [])
                used_babel = True if len(ds_npz_contents) > 0 else False

            if len(ds_npz_contents) == 0:
                subset_dir = osp.join(amass_npz_dir, ds_name)
                ds_npz_contents = glob(osp.join(subset_dir, '*/*_stageii.npz'))

            npz_fnames[split_name].extend(ds_npz_contents)
            logger.debug(
                f'Body: {len(ds_npz_contents):05d} sequences found from AMASS subset {ds_name}. used_babel = {used_babel}')

    return npz_fnames


class MANO():
    def __init__(self):
        raise NotImplementedError('This functionality is not released for current SOMA.')

def face_populate_source():
    raise NotImplementedError('This functionality is not released for current SOMA.')

def hand_populate_source():
    raise NotImplementedError('This functionality is not released for current SOMA.')



def betas_populate_source(amass_npz_dir, amass_splits, gender, num_betas=10):
    '''

    :param datasets: which datasets to get shapes from
    :param amass_npz_dir: amass directory with npz files per subject of each dataset
    :param outpath: path to betas.pt file
    :return:
    '''
    assert gender in ['male', 'female', 'neutral']

    data_betas = {k: [] for k in amass_splits.keys()}
    for split_name in amass_splits.keys():
        if amass_splits[split_name] is None: continue
        for ds_name in amass_splits[split_name]:

            npz_fnames = glob(osp.join(amass_npz_dir, ds_name, f'*/{gender}*.npz'))

            if len(npz_fnames) == 0:
                logger.error(f'no amass betas found at {osp.join(amass_npz_dir, ds_name)}')
            cur_beta_count = 0
            for npz_fname in npz_fnames:
                cdata = np.load(npz_fname, allow_pickle=True)

                if str(cdata['gender'].astype(np.str)) == gender:
                    data_betas[split_name].append(cdata['betas'][:num_betas])
                    cur_beta_count += 1
            logger.debug(f'Betas: {cur_beta_count:04d} shapes chosen from  {ds_name} for {split_name}')
    assert len(data_betas) > 0

    return {k: np.stack(data_betas[k]) if data_betas[k] else None for k in amass_splits.keys()}
