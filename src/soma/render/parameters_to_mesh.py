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

import cv2
import numpy as np
import seaborn as sns
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel

sns.set_theme()

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import makepath

from body_visualizer.mesh.psbody_mesh_sphere import points_to_spheres
from body_visualizer.mesh.psbody_mesh_cube import points_to_cubes

from psbody.mesh import Mesh
from human_body_prior.tools.rotation_tools import rotate_points_xyz

from moshpp.mosh_head import MoSh
from loguru import logger
from moshpp.tools.mocap_interface import MocapSession

from soma.render.blender_tools import prepare_render_cfg


def convert_to_mesh_once(cfg):
    cfg = prepare_render_cfg(**cfg)

    logger.info(f'Preparing mesh files for: {cfg.mesh.mosh_stageii_pkl_fnames}')
    logger.info(f'dirs.mesh_out_dir: {cfg.dirs.mesh_out_dir}')

    datas = {}
    selected_frames = None
    time_length = None
    for mosh_stageii_pkl_fname in cfg.mesh.mosh_stageii_pkl_fnames:
        mosh_id = '/'.join(mosh_stageii_pkl_fname.replace('.pkl', '').split('/')[-2:])

        datas[mosh_id] = {}

        mosh_result = MoSh.load_as_amass_npz(mosh_stageii_pkl_fname, include_markers=True)

        # logger.info(mosh_result.keys())

        num_betas = len(mosh_result['betas']) if 'betas' in mosh_result else 10
        num_dmpls = None if cfg.mesh.enable_dmpl and 'dmpls' in mosh_result else None
        surface_model_type = mosh_result['surface_model_type']
        gender = mosh_result['gender']
        surface_model_fname = osp.join(cfg.dirs.support_base_dir, surface_model_type, gender, 'model.npz')
        assert osp.exists(surface_model_fname), FileExistsError(surface_model_fname)
        if num_dmpls:
            dmpl_fname = osp.join(cfg.dirs.support_base_dir, surface_model_type, gender, 'dmpl.npz')
            assert osp.exists(dmpl_fname), FileExistsError(dmpl_fname)
        else:
            dmpl_fname = None

        num_expressions = len(mosh_result['expression']) if 'expression' in mosh_result else None

        # Todo add object model here
        sm = BodyModel(bm_fname=surface_model_fname,
                       num_betas=num_betas,
                       num_expressions=num_expressions,
                       num_dmpls=num_dmpls,
                       dmpl_fname=dmpl_fname)

        datas[mosh_id]['faces'] = c2c(sm.f)

        # selected_frames = range(0, 10, step_size)
        if selected_frames is None:
            time_length = len(mosh_result['trans'])
            selected_frames = range(0, time_length, cfg.mesh.ds_rate)

        assert time_length == len(mosh_result['trans']), \
            ValueError(
                f'All mosh sequences should have same length. {mosh_stageii_pkl_fname} '
                f'has {len(mosh_result["trans"])} != {time_length}')

        datas[mosh_id]['markers'] = mosh_result['markers'][selected_frames]
        datas[mosh_id]['labels'] = mosh_result['labels']
        # todo: add the ability to have a control on marker colors here

        datas[mosh_id]['num_markers'] = mosh_result['markers'].shape[1]

        if 'betas' in mosh_result:
            mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=time_length, axis=0)

        body_keys = ['betas', 'trans', 'pose_body', 'root_orient', 'pose_hand']

        if 'v_template' in mosh_result:
            mosh_result['v_template'] = np.repeat(mosh_result['v_template'][None], repeats=time_length, axis=0)
            body_keys += ['v_template']
        if num_expressions == 'smplx':
            body_keys += ['expression']
        if num_dmpls:
            body_keys += ['dmpls']

        surface_parms = {k: torch.Tensor(v[selected_frames]) for k, v in mosh_result.items() if k in body_keys}

        datas[mosh_id]['mosh_bverts'] = c2c(sm(**surface_parms).v)

        if cfg.render.show_markers:
            datas[mosh_id]['marker_meta'] = mosh_result['marker_meta']

        num_verts = sm.init_v_template.shape[1]
        datas[mosh_id]['body_color_mosh'] = np.ones([num_verts, 3]) * \
                                            (cfg.mesh.colors.get(mosh_id, cfg.mesh.colors.default))

        first_frame_rot = cv2.Rodrigues(mosh_result['root_orient'][0].copy())[0]
        datas[mosh_id]['theta_z_mosh'] = np.rad2deg(np.arctan2(first_frame_rot[1, 0], first_frame_rot[0, 0]))

    for t, fId in enumerate(selected_frames):
        body_mesh = None
        marker_mesh = None

        for mosh_id, data in datas.items():

            cur_body_verts = rotate_points_xyz(data['mosh_bverts'][t][None],
                                               np.array([0, 0, -data['theta_z_mosh']]).reshape(-1, 3))
            cur_body_verts = rotate_points_xyz(cur_body_verts, np.array([-90, 0, 0]).reshape(-1, 3))[0]

            cur_body_mesh = Mesh(cur_body_verts, data['faces'], vc=data['body_color_mosh'])
            body_mesh = cur_body_mesh if body_mesh is None else body_mesh.concatenate_mesh(cur_body_mesh)

            if cfg.render.show_markers:

                nonan_mask = MocapSession.marker_availability_mask(data['markers'][t:t + 1])[0]
                marker_radius = np.array([
                    cfg.mesh.marker_radius.get(data['marker_meta']['marker_type'][m],
                                               cfg.mesh.marker_radius.default) if m in data['marker_meta'][
                        'marker_type']
                    else cfg.mesh.marker_radius.default
                    for m in data['labels']])
                ghost_mask = np.array([l == 'nan' for l, valid in zip(data['labels'], nonan_mask) if valid],
                                      dtype=np.bool)
                if cfg.mesh.marker_color.style == 'superset':
                    marker_colors = np.array([data['marker_meta']['marker_colors'][m]
                                              if m in data['marker_meta']['marker_type']
                                              else cfg.mesh.marker_color.default for m in data['labels']])
                else:
                    marker_colors = np.array([Color(cfg.mesh.marker_color.style).rgb for _ in data['labels']])

                cur_marker_verts = rotate_points_xyz(data['markers'][t][None],
                                                     np.array([0, 0, -data['theta_z_mosh']]).reshape(-1, 3))

                cur_marker_verts = rotate_points_xyz(cur_marker_verts, np.array([-90, 0, 0]).reshape(-1, 3))[0]
                if cfg.mesh.marker_color.style == 'black':
                    cur_marker_mesh = points_to_spheres(cur_marker_verts[nonan_mask],
                                                    radius=marker_radius[nonan_mask],
                                                    point_color=marker_colors[nonan_mask])
                else:
                    cur_marker_mesh = points_to_spheres(cur_marker_verts[nonan_mask][~ghost_mask],
                                                    radius=marker_radius[nonan_mask][~ghost_mask],
                                                    point_color=marker_colors[nonan_mask][~ghost_mask])
                if ghost_mask.sum() and cfg.mesh.marker_color.style != 'black':
                    try:
                        cur_ghost_mesh = points_to_cubes(cur_marker_verts[nonan_mask][ghost_mask],
                                                         radius=marker_radius[nonan_mask][ghost_mask],
                                                         point_color=np.ones([ghost_mask.sum(), 3]) * [0.83, 1,
                                                                                                       0])  # yellow cube
                        cur_marker_mesh = cur_marker_mesh.concatenate_mesh(cur_ghost_mesh)
                    except:
                        pass

                marker_mesh = cur_marker_mesh if marker_mesh is None else marker_mesh.concatenate_mesh(cur_marker_mesh)

        body_mesh.write_obj(makepath(cfg.dirs.mesh_out_dir, 'body_mesh', f'{fId:05d}.obj', isfile=True))
        if cfg.render.show_markers:
            cur_marker_mesh.write_ply(makepath(cfg.dirs.mesh_out_dir, 'marker_mesh', f'{fId:05d}.ply', isfile=True))

        if cfg.render.render_only_one_image: break

    logger.info(f'Created {cfg.dirs.mesh_out_dir}')

    return
