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
#
"""
Given an already created synthetic data set, produce another extended one with new marker layout.
"""

import glob
import os.path as osp
from copy import deepcopy
from pathlib import Path
from typing import OrderedDict as OrderedDictType
from typing import Union, List, Dict

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.data.dataloader import VPoserDS
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import create_list_chunks
from human_body_prior.tools.omni_tools import makepath
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.edit_tools import randomize_marker_layout_vids
from moshpp.mosh_head import MoSh


def compute_vertex_normal_batched(vertices: torch.Tensor, indices: torch.Tensor):
    from pytorch3d.structures import Meshes
    return Meshes(verts=vertices,
                  faces=indices.expand(len(vertices), -1, -1)).verts_normals_packed().view(-1, vertices.shape[1], 3)


def drop_dict_as_pt(data_dict: Dict[str, torch.Tensor],
                    out_dir: Union[str, Path],
                    aggregate_method: str = 'concatenate'):
    v = None
    for k, v in data_dict.items():
        if aggregate_method == 'concatenate':
            v = np.concatenate(data_dict[k])
        elif aggregate_method == 'stack':
            v = np.stack(data_dict[k], axis=0)

        outfname = makepath(out_dir, '%s.pt' % k, isfile=True)
        # print('{} {} size {}, {}'.format(type(v), k, v[0].shape, v[100].shape))

        if osp.exists(outfname): continue
        torch.save(torch.from_numpy(np.asarray(v)), outfname)
    if v is not None:
        logger.debug(f'Dumped {len(v)} data points as pytorch pt files at: {out_dir}')


def dataset_exists(dataset_dir: Union[str, Path], split_names: List[str] = None) -> bool:
    """
    This function checks whether a valid SOMA dataset directory exists at a location

    Args:
        dataset_dir:
        split_names:

    Returns:

    """
    if dataset_dir is None: return False
    if split_names is None:
        split_names = ['train', 'vald']

    done = []
    for split_name in split_names:
        for k in ['root_orient', 'pose_body', 'betas', 'trans']:
            out_fname = osp.join(dataset_dir, split_name, f'{k}.pt')
            done.append(osp.exists(out_fname))
    return np.all(done)


def sort_markers_like_superset(markers: np.ndarray, labels: list, superset_labels: list):
    """
    Given superset labels will adjust their order to superset labels and leave non-existing markers as zero
    Args:
        markers:
        labels:
        superset_labels:

    Returns:

    """
    num_superset_labels = len(superset_labels)
    time_length = len(markers)
    markers_rearranged = np.zeros([time_length, num_superset_labels, 3])

    for t in range(time_length):
        mocap_lids = [labels[t].index(l) for l in superset_labels if l in labels[t]]
        superset_lids = [superset_labels.index(l) for l in superset_labels if l in labels[t]]
        assert len(mocap_lids) != 0
        markers_rearranged[t, superset_lids] = markers[t][mocap_lids]
    return markers_rearranged


def prepare_real_marker_from_mosh_stageii_pkls(marker_vids: OrderedDictType[str, int],
                                               ds_cfg: OrderedDictType):
    superset_labels = list(marker_vids.keys())

    wanted_fields = ['betas', 'expression', 'markers', 'pose_body',
                     'pose_eye', 'pose_hand', 'pose_jaw', 'root_orient', 'trans']

    def run(mosh_stageii_pkl_fnames):
        out_put_data = {}

        for mosh_stageii_pkl_fname in tqdm(mosh_stageii_pkl_fnames):
            # breakpoint()  # todo replace with proper code
            # data = read_mosh_pkl(mosh_stageii_pkl_fname, superset_labels)
            data = MoSh.load_as_amass_npz(stageii_pkl_data_or_fname=mosh_stageii_pkl_fname, include_markers=True)
            data['markers'] = sort_markers_like_superset(data['markers_obs'], data['labels_obs'], superset_labels)

            if data['gender'] != ds_cfg.gender: continue

            time_length = len(data['markers'])

            if 'betas' in data:
                data['betas'] = np.repeat(data['betas'][:ds_cfg.num_betas][None], repeats=time_length, axis=0)

            # if ds_cfg.animate_face:
            #     assert 'expression' in data, ValueError('face animation is enabled yet real marker data doesnt have expressions')
            # breakpoint()
            if ds_cfg.animate_face:
                if 'expression' in data:
                    data['expression'] = data['expression'][:, :ds_cfg.num_expressions]
                else:
                    data['expression'] = np.zeros([time_length, ds_cfg.num_expressions])
            else:
                for k in ['pose_eye', 'pose_jaw', 'expression']:
                    if k in data: data.pop(k)
            if not ds_cfg.animate_hand and 'pose_hand' in data:
                data.pop('pose_hand')

            ds_rate = max(1, int(data['mocap_frame_rate'] // ds_cfg.unified_frame_rate))

            for k in wanted_fields:
                if k in data:
                    data[k] = data[k][::ds_rate]

            for k in wanted_fields:
                if k not in data: continue
                if k not in out_put_data: out_put_data[k] = []
                for tIds in create_list_chunks(range(data[k].shape[0]),
                                               ds_cfg.num_timeseq_frames,
                                               ds_cfg.num_frames_overlap):
                    out_put_data[k].append(c2c(data[k][tIds]).reshape(1, len(tIds), -1).astype(np.float32))

        # breakpoint()

        out_put_data = {k: np.concatenate(v) for k, v in out_put_data.items()}
        out_put_data['data_is_real'] = np.ones(len(out_put_data['trans'])).astype(np.bool)

        logger.debug('real data: {}'.format({k: v.shape for k, v in out_put_data.items()}))
        return out_put_data.copy()
        # return out_put_data

    return run


def put_markers_on_synthetic_body(marker_vids: OrderedDictType[str, int],
                                  marker_type_mask: OrderedDictType[str, np.ndarray],
                                  m2b_dist_array: np.ndarray, surface_model_fname: Union[str, Path],
                                  wrist_markers_on_stick: bool = False,
                                  num_random_vid_ring: int = 0, num_marker_layout_augmentation: int = 1,
                                  enable_rnd_vid_on_face_hands: bool = False, static_props_array: np.ndarray = None):
    wrist_mask = np.array([True if l in ['RIWR', 'ROWR', 'LIWR', 'LOWR'] else False for l in marker_vids.keys()])
    wrist_mask = wrist_mask[None, :, None]
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    marker_layout_randomizer = randomize_marker_layout_vids(marker_vids=marker_vids,
                                                            marker_type_mask=marker_type_mask,
                                                            n_ring=num_random_vid_ring,
                                                            enable_rnd_vid_on_face_hands=enable_rnd_vid_on_face_hands,
                                                            surface_model_fname=surface_model_fname)

    def run(body_ds, ds_cfg):
        assert len(body_ds), ValueError('provided body dataset is empty!')

        dataloader = DataLoader(body_ds, batch_size=32, shuffle=False, num_workers=10, drop_last=False)

        data_dict = {}
        # todo: check on ds_cfg.surface_model_fname
        bm = BodyModel(surface_model_fname,
                       num_betas=ds_cfg.num_betas,
                       num_expressions=ds_cfg.num_expressions).to(comp_device)
        assert bm.model_type == ds_cfg.surface_model_type, \
            ValueError(f'the model type of the body dataset is not the same as current:'
                       ' {bm.model_type} != {ds_cfg.surface_model_type}')

        for body_parms in tqdm(dataloader):
            body_parms = {k: v.view(body_parms['trans'].shape[0] * ds_cfg.num_timeseq_frames, -1) for k, v in
                          body_parms.items()}
            bs = body_parms['trans'].shape[0]
            body_parms['betas'] = body_parms['betas'][:, :ds_cfg.num_betas]
            if ds_cfg.animate_face:
                body_parms['expression'] = body_parms['expression'][:, :ds_cfg.num_expressions]
                body_parms['pose_eye'] = body_parms['expression'].new(np.zeros([bs, 6]))

            with torch.no_grad():

                body = bm(**{k: v.to(comp_device) for k, v in body_parms.items() if k not in ['joints']})
                vertices = body['v'] if isinstance(body, dict) else body.v
                faces = body['f'] if isinstance(body, dict) else body.f
                vn = compute_vertex_normal_batched(vertices, faces)

            cur_m2b_dist = m2b_dist_array.copy()
            if wrist_markers_on_stick:
                cur_m2b_dist[wrist_mask] = np.random.choice([0.0095, 0.039])

            for _ in range(num_marker_layout_augmentation):
                new_vids = np.array(list(marker_layout_randomizer().values()))
                # print(new_vids, vertices.shape, vn.shape, vertices.shape, new_vids.max())
                markers = c2c(vertices[:, new_vids] + vn[:, new_vids] * vn.new(cur_m2b_dist))
                # if np.any(np.isnan(markers)):#, ValueError('NaN value encountered in marker data')
                #     breakpoint()

                if static_props_array is not None:
                    prop_markers = []
                    for fid in range(len(markers)):
                        prob_id = np.random.choice(len(static_props_array))
                        prop_markers.append(static_props_array[prob_id:prob_id + 1])

                    body_parms['prop_markers'] = np.concatenate(prop_markers, axis=0)

                body_parms['markers'] = markers

                for k in body_parms.keys():
                    if k not in data_dict: data_dict[k] = []
                    data_dict[k].append(
                        c2c(deepcopy(body_parms[k])).reshape(bs // ds_cfg.num_timeseq_frames,
                                                             ds_cfg.num_timeseq_frames,
                                                             -1).astype(np.float32))

        results = {k: np.concatenate(v, axis=0) for k, v in data_dict.items()}
        results['data_is_real'] = np.zeros(len(results['trans'])).astype(np.bool)
        logger.debug('synthetic data: {}'.format({k: v.shape for k, v in results.items()}))
        return results.copy()

    return run


def prepare_marker_dataset(marker_dataset_dir: Union[str, Path],
                           superset_fname: Union[str, Path],
                           body_dataset_dir: Union[str, Path],
                           real_marker_amass_splits,
                           amass_pkl_dir: Union[str, Path],
                           wrist_markers_on_stick: bool,
                           use_real_data_for: List[str],
                           use_synt_data_for: List[str],
                           surface_model_fname: Union[str, Path],
                           num_random_vid_ring: int = 0,
                           num_marker_layout_augmentation: int = 1,
                           static_props_array: np.ndarray = None,
                           enable_rnd_vid_on_face_hands: bool = True,
                           babel: Dict[str, List[str]] = None):
    """
    We use this to be able to creat synthetic marker data using different marker layouts

    Args:
        marker_dataset_dir:
        superset_fname:
        body_dataset_dir:
        real_marker_amass_splits:
        amass_pkl_dir:
        wrist_markers_on_stick:
        use_real_data_for:
        use_synt_data_for:
        surface_model_fname:
        num_random_vid_ring:
        num_marker_layout_augmentation:
        static_props_array:
        enable_rnd_vid_on_face_hands:
        babel:

    Returns:

    """
    if dataset_exists(marker_dataset_dir):
        logger.debug(f'Marker dataset already exists at {marker_dataset_dir}')
        return

    log_fname = makepath(marker_dataset_dir, 'dataset.log', isfile=True)
    log_format = "{module}:{function}:{line} -- {level} -- {message}"
    ds_logger_id = logger.add(log_fname, format=log_format, enqueue=True)

    marker_meta = marker_layout_load(superset_fname)
    marker_vids = marker_meta['marker_vids']  # superset should a mapping between labels: str to list of vertex ids
    marker_type_mask = marker_meta['marker_type_mask']
    m2b_dist = np.ones(len(marker_vids)) * 0.0095
    for mask_type, marker_mask in marker_meta['marker_type_mask'].items():
        m2b_dist[marker_mask] = marker_meta['m2b_distance'][mask_type]
    m2b_dist = m2b_dist[None, :, None]

    logger.debug(f'Creating marker dataset with superset {superset_fname} which has {len(marker_vids)} markers.')
    logger.debug(f'Superset marker type distance {dict(marker_meta["m2b_distance"])} meters.')
    if num_random_vid_ring > 0:
        logger.debug(f'Will place markers randomly on {num_random_vid_ring} ring neighbourhood of a vid.')

    body_ds_cfg_fname = osp.join(body_dataset_dir, 'settings.yaml')
    assert osp.exists(body_ds_cfg_fname), FileNotFoundError(body_ds_cfg_fname)
    body_ds_cfg = OmegaConf.load(body_ds_cfg_fname)

    for split_name in ['train', 'vald']:
        # for split_name in ['vald', 'train']:
        body_ds = VPoserDS(dataset_dir=osp.join(body_dataset_dir, split_name))
        assert len(body_ds), ValueError(f'No body dataset found at: {osp.join(body_dataset_dir, split_name)}')

        if dataset_exists(marker_dataset_dir, [split_name]): continue
        logger.debug(f'Preparing data files for split {split_name}')

        if use_synt_data_for is not None and split_name in use_synt_data_for:
            logger.debug(
                f'Preparing synthetic marker data for split {split_name} from '
                f'#{len(body_ds)} body parameters corresponding to datasets {body_ds_cfg.amass_splits[split_name]}')
            synt_data_dict = put_markers_on_synthetic_body(marker_vids=marker_vids,
                                                           marker_type_mask=marker_type_mask,
                                                           m2b_dist_array=m2b_dist,
                                                           wrist_markers_on_stick=wrist_markers_on_stick,
                                                           num_random_vid_ring=num_random_vid_ring,
                                                           num_marker_layout_augmentation=num_marker_layout_augmentation,
                                                           enable_rnd_vid_on_face_hands=enable_rnd_vid_on_face_hands,
                                                           surface_model_fname=surface_model_fname,
                                                           static_props_array=static_props_array)(body_ds, body_ds_cfg)
            logger.debug(f'#{len(synt_data_dict["trans"])} synthetic data points created')
        else:
            logger.debug(
                f'Not using synthetic data for split {split_name} since use_synt_data_for ({use_synt_data_for}) does not include this split')
            synt_data_dict = {}

        if use_real_data_for and split_name in use_real_data_for:
            logger.debug('To be able to use real data for training you need to obtain real mocap '
                         'markers of AMASS from the respective original datasets and mosh them')
            mosh_stageii_pkl_fnames = []
            for ds_name in real_marker_amass_splits[split_name]:
                cur_mosh_stageii_pkl_fnames, use_babel = [], False
                if babel and ds_name in babel:
                    cur_mosh_stageii_pkl_fnames = [fname.replace('.npz', '.pkl') for fname in babel[ds_name]]
                    if cur_mosh_stageii_pkl_fnames: use_babel = True
                else:
                    cur_mosh_stageii_pkl_fnames = glob.glob(osp.join(amass_pkl_dir, ds_name, '*/*_stageii.pkl'))

                logger.debug(
                    f'Obtained {len(cur_mosh_stageii_pkl_fnames):05d} sequences for real mocap from AMASS subset {ds_name}. used_babel = {use_babel}')

                mosh_stageii_pkl_fnames.extend(cur_mosh_stageii_pkl_fnames)

            logger.debug(
                f"Preparing real marker data for split {split_name} from #{len(mosh_stageii_pkl_fnames)} mosh_stageii_pkl_fnames {real_marker_amass_splits[split_name]}.")
            real_data_dict = prepare_real_marker_from_mosh_stageii_pkls(marker_vids, body_ds_cfg)(
                mosh_stageii_pkl_fnames)
            logger.debug(
                f'#{len(real_data_dict["trans"])} real data points extracted from #{len(mosh_stageii_pkl_fnames)} mosh_stageii_pkl_fnames')
        else:
            logger.debug(
                f'Not using real data for split {split_name} since use_real_data_for ({use_real_data_for}) does not have this split name')
            real_data_dict = {}

        data_keys = list(set(list(synt_data_dict.keys()) + list(real_data_dict.keys())))
        data_dict = {k: [] for k in data_keys}
        for k in data_keys:
            # add key to data showing real and synthetic data
            if k in synt_data_dict: data_dict[k].append(synt_data_dict[k])
            if k in real_data_dict: data_dict[k].append(real_data_dict[k])

        drop_dict_as_pt(data_dict=data_dict, out_dir=makepath(marker_dataset_dir, split_name))

    save_cfg = OmegaConf.create({
        'real_marker_amass_splits': real_marker_amass_splits,
        'synthetic_body_amass_splits': body_ds_cfg.amass_splits,
        'superset_fname': superset_fname,
        'wrist_markers_on_stick': wrist_markers_on_stick,
        'use_real_data_for': use_real_data_for,
        'use_synt_data_for': use_synt_data_for,
        'body_dataset_dir': body_dataset_dir,
        'surface_model_fname': surface_model_fname,
        'num_random_vid_ring': num_random_vid_ring,
        'enable_rnd_vid_on_face_hands': enable_rnd_vid_on_face_hands,
        'num_marker_layout_augmentation': num_marker_layout_augmentation,
        'babel': babel,
        'num_prop_marker_max': static_props_array.shape[1] if static_props_array is not None else 0,
        'unified_frame_rate': body_ds_cfg.unified_frame_rate,
        'num_hand_var_perseq': body_ds_cfg.num_hand_var_perseq,
        'num_betas': body_ds_cfg.num_betas,
        'num_expressions': body_ds_cfg.num_expressions,
        'gender': body_ds_cfg.gender,
        'amass_pkl_dir': amass_pkl_dir,
        'num_timeseq_frames': body_ds_cfg.num_timeseq_frames,
        'num_frames_overlap': body_ds_cfg.num_frames_overlap,
    })
    OmegaConf.save(config=save_cfg, f=makepath(marker_dataset_dir, 'settings.yaml', isfile=True))

    logger.debug(f'marker_dataset_dir: {marker_dataset_dir}')
    logger.remove(ds_logger_id)
    return marker_dataset_dir
