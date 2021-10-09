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
import sys
from os import path as osp

import bpy
from human_body_prior.tools.omni_tools import get_support_data_dir
from loguru import logger
from omegaconf import OmegaConf, DictConfig

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
from soma.tools.parallel_tools import run_parallel_jobs


def render_mosh_once(render_cfg):
    from soma.render.parameters_to_mesh import convert_to_mesh_once
    from soma.render.mesh_to_video_standard import create_video_from_mesh_dir
    convert_to_mesh_once(render_cfg)
    create_video_from_mesh_dir(render_cfg)


def prepare_render_cfg(*args, **kwargs) -> DictConfig:
    setup_mosh_omegaconf_resolvers()

    if not OmegaConf.has_resolver('resolve_out_basename'):
        OmegaConf.register_new_resolver('resolve_out_basename',
                                        lambda mosh_stageii_pkl_fnames:
                                        mosh_stageii_pkl_fnames[0].split('/')[-1].replace('_stageii.pkl', ''))

    if not OmegaConf.has_resolver('resolve_subject_action_name'):
        OmegaConf.register_new_resolver('resolve_subject_action_name',
                                        lambda mosh_stageii_pkl_fnames: mosh_stageii_pkl_fnames[0].split('/')[-2])

    if not OmegaConf.has_resolver('resolve_out_ds_name'):
        OmegaConf.register_new_resolver('resolve_out_ds_name',
                                        lambda mosh_stageii_pkl_fnames: mosh_stageii_pkl_fnames[0].split('/')[-3])

    app_support_dir = get_support_data_dir(__file__)
    base_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/render_conf.yaml'))

    override_cfg_dotlist = [f'{k}={v}' for k, v in kwargs.items()]
    override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

    return OmegaConf.merge(base_cfg, override_cfg)


def setup_scene(cfg):
    assert osp.exists(cfg.render.blender_fname), FileNotFoundError(cfg.render.blender_fname)
    logger.info(f'Opening scene from provided blend file {cfg.render.blender_fname}')
    bpy.ops.wm.open_mainfile(filepath=cfg.render.blender_fname)

    for scene_name, scene in bpy.data.scenes.items():
        logger.info(f'Setting up scene: {scene_name}')
        # scene = bpy.data.scenes['Scene']

        if cfg.render.render_engine.lower() == 'eevee':
            scene.render.engine = 'BLENDER_EEVEE'
            scene.eevee.taa_render_samples = cfg.render.num_samples.eevee
        else:
            scene.render.engine = 'CYCLES'
            scene.cycles.samples = cfg.render.num_samples.cycles

        if cfg.render.resolution.change_from_blend:
            scene_resolution = cfg.render.resolution.get(scene_name, cfg.render.resolution.default)
            scene.render.resolution_x = scene_resolution[0]
            scene.render.resolution_y = scene_resolution[1]

        scene.render.image_settings.color_mode = 'RGBA'
    plane = [obj for collect in bpy.data.collections for obj in collect.all_objects if obj.name in ['Plane', ]][0]
    if not cfg.render.floor.enable:
        bpy.ops.object.delete({"selected_objects": [plane]})
    else:
        plane.location = cfg.render.floor.plane_location


def make_blender_silent():
    # Silence console output of bpy.ops.render.render by redirecting stdout to /dev/null
    sys.stdout.flush()
    old = os.dup(1)
    os.close(1)
    os.open(os.devnull, os.O_WRONLY)


def render_mosh_stageii(mosh_stageii_pkl_fnames, render_cfg=None, parallel_cfg=None, **kwargs):
    if parallel_cfg is None: parallel_cfg = {}
    if render_cfg is None: render_cfg = {}

    fname_filter = kwargs.get('fname_filter', None)

    app_support_dir = get_support_data_dir(__file__)
    base_parallel_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/parallel_conf/blender_parallel.yaml'))

    total_jobs = []
    for mosh_stageii_pkl_fname in mosh_stageii_pkl_fnames:
        assert mosh_stageii_pkl_fname.endswith('_stageii.pkl')
        if fname_filter:
            if not sum([i in mosh_stageii_pkl_fname for i in fname_filter]): continue
        job = render_cfg.copy()
        job.update({
            'mesh.mosh_stageii_pkl_fnames': [mosh_stageii_pkl_fname],
        })

        render_cfg = prepare_render_cfg(**job)

        if osp.exists(render_cfg.dirs.mp4_out_fname): continue
        total_jobs.append(job.copy())

    parallel_cfg = OmegaConf.merge(base_parallel_cfg, OmegaConf.create(parallel_cfg))

    run_parallel_jobs(func=render_mosh_once, jobs=total_jobs, parallel_cfg=parallel_cfg)
