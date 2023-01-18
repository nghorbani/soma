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
import shutil
import sys
from datetime import datetime as dt

import numpy as np
import torch
from human_body_prior.tools.model_loader import load_model
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import create_list_chunks
from human_body_prior.tools.omni_tools import get_support_data_dir
from human_body_prior.tools.omni_tools import makepath
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from moshpp.mosh_head import setup_mosh_omegaconf_resolvers
from moshpp.tools.mocap_interface import MocapSession
from soma.models.soma_model import SOMA
from soma.tools.permutation_tools import batch_shuffle, batched_index_select


def create_labels_count(markers, labels_perframe):
    labels_count = {}
    zero_mask = (markers == 0).sum(-1) == 3

    for tracklet_id in range(labels_perframe.shape[-1]):
        labels_count[tracklet_id] = {}
        # marker_labels = labels_perframe[:, tracklet_id].tolist()
        # for l in set(marker_labels):
        #     if l == 'nan': continue
        #     labels_count[tracklet_id][l] = marker_labels.count(l)
        for t in range(labels_perframe.shape[0]):
            if zero_mask[t, tracklet_id]: continue
            l = labels_perframe[t, tracklet_id]
            if l not in labels_count[tracklet_id]: labels_count[tracklet_id][l] = 0
            labels_count[tracklet_id][l] += 1
    return labels_count


class SOMAMoCapPointCloudLabeler(object):
    """
    SOMA labels 3D points captured during a marker based optical motion capture session

    """

    # def __init__(self, mocap_fname, soma_model, soma_ps, tracker_ps, logger=None):
    def __init__(self, **kwargs) -> None:

        super(SOMAMoCapPointCloudLabeler, self).__init__()

        from moshpp.marker_layout.edit_tools import marker_layout_load
        from moshpp.marker_layout.labels_map import general_labels_map

        self.rt_cfg = SOMAMoCapPointCloudLabeler.prepare_cfg(**kwargs)  # runtime configs

        logger.remove()
        if self.rt_cfg.verbosity > 0:
            makepath(self.rt_cfg.dirs.log_fname, isfile=True)

            log_format = f"{{module}}:{{function}}:{{line}} -- " \
                         f"{self.rt_cfg.soma.expr_id} -- {self.rt_cfg.soma.data_id} -- {self.rt_cfg.mocap.ds_name} -- " \
                         f"{self.rt_cfg.mocap.subject_name} -- {self.rt_cfg.mocap.basename} -- {{message}}"
            logger.add(self.rt_cfg.dirs.log_fname, format=log_format, enqueue=True)
            logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

        logger.info(f'Starting SOMA on {self.rt_cfg.mocap.fname}')

        self.soma_model, self.model_cfg = load_model(self.rt_cfg.soma.expr_dir, model_code=SOMA,
                                                     remove_words_in_model_weights='soma_model.',
                                                     model_cfg_override={
                                                         'dirs.work_base_dir': self.rt_cfg.dirs.work_base_dir,
                                                         'dirs.support_base_dir': self.rt_cfg.dirs.support_base_dir
                                                        }
                                                     )

        assert self.rt_cfg.soma.expr_id == self.model_cfg.soma.expr_id
        assert self.rt_cfg.soma.data_id == self.model_cfg.soma.data_id

        mocap = MocapSession(mocap_fname=self.rt_cfg.mocap.fname,
                             mocap_unit=self.rt_cfg.mocap.unit,
                             mocap_rotate=self.rt_cfg.mocap.rotate,
                             ignore_stared_labels=False
                             )

        self.mocap_frame_rate = mocap.frame_rate

        self.orig_points_scaled = mocap.markers

        # to compute scaled_dot_product_attention span
        # todo: can we avoid these?
        if self.rt_cfg.retain_model_debug_details and 'labels_perframe' in mocap._marker_data:
            orig_mocap_labels_perframe = mocap._marker_data['labels_perframe']

        # Todo: check out if evaluation with and without marker array id yields the same results
        mocap_length = len(mocap)
        origin_mocap_nonzero_mask = MocapSession.marker_availability_mask(mocap.markers)
        max_nonzero_point_perframe = origin_mocap_nonzero_mask.sum(-1).max()
        orig_mocap_array_id = np.arange(np.prod(mocap.markers.shape[:-1])).reshape(mocap_length, -1)
        self.points_compressed = np.zeros([mocap_length, max_nonzero_point_perframe, 3], dtype=np.float)
        self.mocap_array_id = np.ones([mocap_length, max_nonzero_point_perframe], dtype=np.int) * np.nan
        self.point_orig_labels = np.empty([mocap_length, max_nonzero_point_perframe]).astype('<U4')

        for t in range(mocap_length):
            assert origin_mocap_nonzero_mask[t].sum() <= max_nonzero_point_perframe
            self.points_compressed[t, :origin_mocap_nonzero_mask[t].sum()] = mocap.markers[
                t, origin_mocap_nonzero_mask[t]]
            self.mocap_array_id[t, :origin_mocap_nonzero_mask[t].sum()] = orig_mocap_array_id[
                t, origin_mocap_nonzero_mask[t]]
            if self.rt_cfg.retain_model_debug_details:
                self.point_orig_labels[t, :origin_mocap_nonzero_mask[t].sum()] = \
                    orig_mocap_labels_perframe[t, origin_mocap_nonzero_mask[t]]
                self.point_orig_labels[t, origin_mocap_nonzero_mask[t].sum():] = 'nan'

        percent_avail_points = origin_mocap_nonzero_mask.sum() / np.prod(origin_mocap_nonzero_mask.shape)
        logger.info(
            f'loaded mocap points. #markers.shape = {mocap.markers.shape}, '
            f'frame_rate: {mocap.frame_rate} and available points {percent_avail_points * 100:.2f}%')
        logger.info(f'Maximum number of non-zero points per-frame: {max_nonzero_point_perframe}')

        self.superset_fname = self.model_cfg.data_parms.marker_dataset.superset_fname
        self.superset_meta = marker_layout_load(self.superset_fname, labels_map=general_labels_map)
        self.superset_labels = np.array(list(self.superset_meta['marker_colors'].keys()))

        self.num_labels = len(self.superset_labels)
        # self.num_labels = OrderedDict({k:np.sum(v) for k,v in superset_meta['marker_type_mask'].items()})
        #
        logger.info(f'SOMA is trained for #{self.num_labels - 1} markers of layout: {self.superset_fname}')

        self.comp_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.soma_model.to(self.comp_device)
        self.soma_model.eval()

        src_subject_gender_fname = osp.join(osp.dirname(self.rt_cfg.mocap.fname), 'settings.json')
        if osp.exists(src_subject_gender_fname):
            dst_subject_gender_fname = osp.join(osp.dirname(self.rt_cfg.dirs.mocap_out_fname), 'settings.json')
            if not osp.exists(dst_subject_gender_fname):
                shutil.copy2(src_subject_gender_fname, makepath(dst_subject_gender_fname, isfile=True))

    @staticmethod
    def prepare_cfg(**kwargs) -> DictConfig:
        from soma.train.soma_trainer import create_soma_data_id
        setup_mosh_omegaconf_resolvers()
        if not OmegaConf.has_resolver('resolve_soma_data_id'):
            OmegaConf.register_new_resolver('resolve_soma_data_id',
                                            lambda num_occ_max, num_ghost_max, limit_real_data, limit_synt_data:
                                            create_soma_data_id(num_occ_max, num_ghost_max, limit_real_data,
                                                                limit_synt_data))
        if not OmegaConf.has_resolver('resolve_soma_runtime_work_dir'):
            OmegaConf.register_new_resolver('resolve_soma_runtime_work_dir',
                                            lambda tracklet_labeling_enable:
                                            'soma_labeled_mocap' + ('_tracklet' if tracklet_labeling_enable else ''))

        app_support_dir = get_support_data_dir(__file__)
        base_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/soma_run_conf.yaml'))

        override_cfg_dotlist = [f'{k}={v}' for k, v in kwargs.items()]
        override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

        return OmegaConf.merge(base_cfg, override_cfg)

    def apply_soma(self):

        # we will pass only the non-zero values to soma
        batch_size = self.rt_cfg.batch_size
        logger.info(f'Applying SOMA on mocap with batch_size = {batch_size}')

        mocap_length = self.points_compressed.shape[0]
        max_nonzero_point_perframe = self.points_compressed.shape[1]

        label_confidence = np.zeros([mocap_length, max_nonzero_point_perframe, len(self.superset_labels)])

        run_windows_batches = create_list_chunks(np.arange(mocap_length), batch_size, 0, cut_smaller_batches=False)

        if self.model_cfg.model_parms.labeler.enable_transformer:
            all_attention_weights = []

        runtime = 0
        for run_windows_batch_id, frame_ids in tqdm(enumerate(run_windows_batches), total=len(run_windows_batches)):
            points = torch.from_numpy(self.points_compressed[frame_ids]).type(torch.float).to(self.comp_device)

            with torch.no_grad():
                points_permuted, inv_map = batch_shuffle(points)

                process_start_time = dt.now().replace(microsecond=0)

                soma_results = self.soma_model(pts=points_permuted)
                soma_results['label_confidence'] = batched_index_select(soma_results['label_confidence'], 1, inv_map)

                process_end_time = dt.now().replace(microsecond=0)

                runtime += (process_end_time - process_start_time).total_seconds()

            label_confidence[frame_ids] = c2c(soma_results['label_confidence'])
            if self.model_cfg.model_parms.labeler.enable_transformer and 'attention_weights' in soma_results:
                permuted_attention_weights = soma_results['attention_weights']
                permuted_attention_weights = batched_index_select(permuted_attention_weights, 3, inv_map)
                attention_weights = batched_index_select(permuted_attention_weights, 4, inv_map)
                all_attention_weights.append(c2c(attention_weights))

        soma_fps = None if runtime == 0 else mocap_length / runtime

        logger.info(f'SOMA was performing at {soma_fps if soma_fps else "inf"} Hz.')

        results = {'label_confidence': label_confidence, 'soma_fps': soma_fps}

        results.update({'point_orig_labels': self.point_orig_labels,
                        # 'points_compressed': self.points_compressed,
                        })
        if self.model_cfg.model_parms.labeler.enable_transformer and len(all_attention_weights):
            results.update({'attention_weights': np.concatenate(all_attention_weights, axis=0)})

        return results

    @staticmethod
    def create_mocap(points, points_label_id, superset_labels, keep_nan_points=True, remove_zero_trajectories=True):
        """
        given a time sequence of point clouds and soma labels, create a valid mocap
        Parameters
        ----------
        points: Txnum_pointsx3: point 3D position
        points_label_id: Txnum_points: the predicted label_id for each point
        points_label_conf: Txnum_points: the confidence in the selected label
        mocap_array_id: the id in the mocap array. to be used later for performance metrics
        superset_labels: num_markers

        Returns
        -------
        dict
            markers: Tx(num_superset_labels-1)x3: no nan
            labels: num_superset_labels-1
            # labels_conf: Txnum_superset_labels-1
        """

        mocap_length = points.shape[0]
        nan_label_id = np.where(superset_labels == 'nan')[0][0]

        nan_markers = []

        markers = np.zeros([mocap_length, len(superset_labels) - 1, 3], dtype=np.float)  # not including nan label
        # label_conf = np.zeros([mocap_length, len(superset_labels) - 1])  # not including nan label

        for t in range(mocap_length):
            cur_nan_markers = []

            for label_id, label_name in enumerate(superset_labels):

                label_mask = points_label_id[t] == label_id

                if label_mask.sum() == 0:
                    continue  # no label found
                elif label_mask.sum() > 1 or label_id == nan_label_id:  # keep the trash data
                    cur_non_zero_mask = (points[t, label_mask] == 0).sum(-1) < 3
                    if cur_non_zero_mask.sum() > 0:
                        cur_nan_markers.append(points[t, label_mask][cur_non_zero_mask])
                    continue

                markers[t, label_id] = points[t, label_mask]
                # label_conf[t, label_id] = points_label_conf[t, label_mask]

            nan_markers.append(np.concatenate(cur_nan_markers) if len(cur_nan_markers) else [])

        labels = superset_labels[:-1]  # [accept_trajectory_mask]
        label_ids = np.arange(len(labels))
        max_num_nan = max([len(i) for i in nan_markers])
        if keep_nan_points and max_num_nan > 0:
            all_nan_markers = np.zeros([mocap_length, max_num_nan, 3], dtype=np.float)  # not including nan label
            for t, m in enumerate(nan_markers):
                if len(m) == 0: continue
                all_nan_markers[t, :len(m)] = m

            markers = np.concatenate([markers, all_nan_markers], axis=1)
            # label_conf = np.concatenate([label_conf, all_nan_confidence], axis=1)
            label_ids = np.concatenate([label_ids, [nan_label_id] * max_num_nan])
            labels = np.concatenate([labels, [superset_labels[nan_label_id]] * max_num_nan])
        if remove_zero_trajectories:
            # drop all zero labels
            non_zero_tracks_mask = (markers == 0).sum(-1).sum(0) < len(markers)
        else:
            non_zero_tracks_mask = np.ones([markers.shape[1]], dtype=np.bool)

        return {'markers': markers[:, non_zero_tracks_mask],
                'labels_perframe': np.repeat(labels[non_zero_tracks_mask][None], repeats=mocap_length, axis=0),
                'labels': labels[non_zero_tracks_mask],
                'label_ids': label_ids[non_zero_tracks_mask],
                'label_ids_perframe': np.repeat(label_ids[non_zero_tracks_mask][None], repeats=mocap_length, axis=0),
                # 'label_conf': label_conf[:, non_zero_tracks_mask],
                }

    def label_perframe(self, label_confidence):
        """
        only take above threshold labels that are disjoint
        """

        # point_label_conf = label_confidence.max(-1)
        points_label_id = label_confidence.argmax(-1)

        labeling_results = self.create_mocap(points=self.points_compressed,
                                             points_label_id=points_label_id,
                                             superset_labels=self.superset_labels,
                                             keep_nan_points=self.rt_cfg.keep_nan_points,
                                             remove_zero_trajectories=self.rt_cfg.remove_zero_trajectories)

        markers, labels = labeling_results['markers'], labeling_results['labels']

        logger.info(f'{len(labels)} labels detected')

        percent_points_labeled = (
                1. - ((markers == 0).sum(-1) == 3).sum() / (len(self.points_compressed) * len(labels)))
        logger.success(f'Per-frame labeling yielded {percent_points_labeled * 100:0.2f}% non-zero mocap.')

        return labeling_results

    def label_tracklets(self, label_confidence):
        """
        only take above threshold labels that are disjoint
        """
        # when a label is assigned to a label less than this threshold reject it
        # reject_label_trajectory_thrs = self.tracker_ps.general.reject_label_trajectory_thrs

        nonan_mask = np.logical_not(np.isnan(self.mocap_array_id))
        mocap_array_id_int = self.mocap_array_id.astype(np.int)

        # point_label_conf = label_confidence.max(-1)
        points_label_id = label_confidence.argmax(-1)

        nan_label_id = len(self.superset_labels) - 1
        assert self.superset_labels[nan_label_id] == 'nan'

        mocap_labels_expanded = np.ones(self.orig_points_scaled.shape[:-1], dtype=np.int) * nan_label_id
        mocap_labels_expanded.reshape(-1)[mocap_array_id_int[nonan_mask]] = points_label_id[nonan_mask]

        tracklet_label_counts = create_labels_count(self.orig_points_scaled, mocap_labels_expanded)

        mocap_labels_expanded_tracked = np.ones(self.orig_points_scaled.shape[:-1], dtype=np.int) * nan_label_id

        tracklet_labels = {l: max(candidates, key=candidates.get) if len(candidates) != 0 else nan_label_id for
                           l, candidates in tracklet_label_counts.items()}

        for tracklet_id, lId in tracklet_labels.items():
            mocap_labels_expanded_tracked[:, tracklet_id] = lId

        points_label_id_tracked = np.ones(self.points_compressed.shape[:-1], dtype=np.int) * nan_label_id
        points_label_id_tracked[nonan_mask] = mocap_labels_expanded_tracked.reshape(-1)[mocap_array_id_int[nonan_mask]]

        # point_label_conf = np.ones_like(points_label_id_tracked)
        # ToDo: if a point label is inside a tracklet with the same label keep the confidence if not plunge it to zero

        labeling_results = self.create_mocap(points=self.points_compressed,
                                             points_label_id=points_label_id_tracked,
                                             superset_labels=self.superset_labels,
                                             keep_nan_points=self.rt_cfg.keep_nan_points,
                                             remove_zero_trajectories=self.rt_cfg.remove_zero_trajectories, )

        markers, labels = labeling_results['markers'], labeling_results['labels']

        logger.info(f'{len(labels)} labels detected')

        # percent_points_labeled = (1. - ((markers[:,:len(self.superset_labels)-1]==0).sum(-1)==3).sum()/(self.mocap_length*(len(self.superset_labels)-1)))
        # logger.info(f'{percent_points_labeled * 100:0.2f}% of the output mocap is non-zero for -- {self.mocap_fname}')

        percent_points_labeled = (
                1. - ((markers == 0).sum(-1) == 3).sum() / (len(self.points_compressed) * len(labels)))
        logger.success(f'Tracklet labeling yielded {percent_points_labeled * 100:0.2f}% non-zero mocap.')

        return labeling_results


def run_soma_once(soma_runtime_cfg):
    from soma.tools.soma_processor import SOMAMoCapPointCloudLabeler

    from moshpp.tools.mocap_interface import write_mocap_c3d
    from loguru import logger
    import pickle

    soma_labeler = SOMAMoCapPointCloudLabeler(**soma_runtime_cfg)

    rt_cfg = soma_labeler.rt_cfg

    # if rt_cfg.retain_model_debug_details:
    #     assert rt_cfg.mocap.ds_name.endswith('___OC_00_G_00_BT_00'), \
    #         ValueError(
    #             'To compute scaled_dot_product_attention span you should prepare a dataset with no noise; i.e. OC_00_G_00_BT_00')

    soma_applied = soma_labeler.apply_soma()
    if rt_cfg.soma.tracklet_labeling.enable:
        results = soma_labeler.label_tracklets(label_confidence=soma_applied['label_confidence'])
    else:
        results = soma_labeler.label_perframe(label_confidence=soma_applied['label_confidence'])

    results['superset_meta'] = soma_labeler.superset_meta
    results['frame_rate'] = soma_labeler.mocap_frame_rate

    debug_details = {
        'mocap_fname': rt_cfg.mocap.fname,
        # 'points': c2c(tracker.mocap_points_orig),
        'soma_rt_cfg': rt_cfg,
        'soma_model_cfg': soma_labeler.model_cfg,
        'soma_fps': soma_applied['soma_fps'],
    }
    if rt_cfg.retain_model_debug_details and 'attention_weights' in soma_applied:
        debug_details.update({'attention_weights': soma_applied['attention_weights'],
                              'point_orig_labels': soma_applied['point_orig_labels'],
                              # 'points':soma_applied['points_compressed'],
                              # 'points_label_confidence': soma_applied['label_confidence']

                              })

    # if rt_cfg.retain_model_debug_details:
    #     debug_details['model_debug_details'] = soma_applied['model_debug_details']

    results.update({'debug_details': debug_details})

    pickle.dump(results, open(rt_cfg.dirs.mocap_out_fname, 'wb'))
    logger.success(f'Created {rt_cfg.dirs.mocap_out_fname}')

    if rt_cfg.save_c3d:
        c3d_out_fname = rt_cfg.dirs.mocap_out_fname.replace('.pkl', '.c3d')
        nan_replaced_labels = [l if l != 'nan' else '*{}'.format(i) for i, l in enumerate(results['labels'])]
        write_mocap_c3d(out_mocap_fname=c3d_out_fname,
                        markers=results['markers'],
                        labels=nan_replaced_labels,
                        frame_rate=soma_labeler.mocap_frame_rate)
        logger.info(f'Created {c3d_out_fname}')

    logger.info('----------------')
