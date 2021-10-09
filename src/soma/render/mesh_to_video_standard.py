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
import os.path as osp
from math import radians

import bpy
from body_visualizer.tools.render_tools import pngs2mp4
from human_body_prior.tools.omni_tools import makepath
from loguru import logger

from soma.render.blender_tools import make_blender_silent
from soma.render.blender_tools import prepare_render_cfg
from soma.render.blender_tools import setup_scene


def run_blender_once(cfg, body_mesh_fname, marker_mesh_fname, png_out_fname):
    make_blender_silent()

    bpy.ops.object.delete({"selected_objects": [obj for colec in bpy.data.collections for obj in colec.all_objects if
                                                obj.name in ['Body', 'Object']]})

    if cfg.render.show_body:
        bpy.ops.import_scene.obj(filepath=body_mesh_fname)

        body = bpy.context.selected_objects[0]

        body.name = 'Body'

        if cfg.render.rotate_body_object_z:
            body.rotation_euler[2] = radians(cfg.render.rotate_body_object_z)

        assert "Body" in bpy.data.materials
        body.active_material = bpy.data.materials['Body']

        # enable quads
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.mode_set(mode='OBJECT')

        # make the surface smooth
        bpy.ops.object.shade_smooth()

        # enable wireframe
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.mark_freestyle_edge(clear=False)
        bpy.ops.object.mode_set(mode='OBJECT')

    if cfg.render.show_markers:
        bpy.ops.import_mesh.ply(filepath=marker_mesh_fname)
        marker_mesh = bpy.context.object
        marker_mesh.name = 'Object'
        # Note: Blender ply importer does not select imported object
        #    object = bpy.context.selected_objects[0]
        marker_mesh.rotation_euler = (radians(90.0), 0.0, 0.0)
        if cfg.render.rotate_body_object_z:
            marker_mesh.rotation_euler[2] = radians(cfg.render.rotate_body_object_z)

        assert "Object" in bpy.data.materials
        marker_mesh.active_material = bpy.data.materials['Object']

        # else:
        #     setup_vertex_color_material(marker_mesh)

    if cfg.render.camera_tracking_mode:
        # Create vertex group so that camera can track the mesh vertices instead if pivot
        body_cam = bpy.data.scenes['Scene'].camera

        # logger.info('creating constrain on vertex_group {} for camera {}'.format(body_part_name, body_cam.name))
        for constraint in body_cam.constraints:
            if cfg.render.camera_tracking_mode == 'body' and cfg.render.show_body:
                constraint.target = body
                body_vertex_group = body.vertex_groups.new(name='body')
                body_vertex_group.add([v.index for v in body.data.vertices], 1.0, 'ADD')
                constraint.subtarget = 'body'
            else:
                constraint.target = marker_mesh
                object_vertex_group = marker_mesh.vertex_groups.new(name='object')
                object_vertex_group.add([v.index for v in marker_mesh.data.vertices], 1.0, 'ADD')
                constraint.subtarget = 'object'

    # Render
    bpy.context.scene.render.filepath = png_out_fname

    # Render
    bpy.ops.render.render(write_still=True)
    if cfg.render.save_final_blend_file:
        bpy.ops.wm.save_as_mainfile(filepath=png_out_fname.replace('.png', '.blend'))

    bpy.ops.object.delete({"selected_objects": [obj for colec in bpy.data.collections for obj in colec.all_objects if
                                                obj.name in ['Body', 'Object']]})

    # # Delete last selected object from scene
    # if ps.show_body:
    #     body.select_set(True)
    # if ps.show_object:
    #     if not ps.show_body:
    #         bpy.ops.import_scene.obj(filepath=body_mesh_fname)
    #     marker_mesh.select_set(True)
    #
    # bpy.ops.object.delete()

    logger.success(f'created {png_out_fname}')

    return


def create_video_from_mesh_dir(cfg):
    cfg = prepare_render_cfg(**cfg)

    makepath(cfg.dirs.png_out_dir)

    setup_scene(cfg)

    logger.debug(f'input mesh dir: {cfg.dirs.mesh_out_dir}')
    logger.debug(f'png_out_dir: {cfg.dirs.png_out_dir}')

    body_mesh_fnames = sorted(glob.glob(os.path.join(cfg.dirs.mesh_out_dir, 'body_mesh', '*.obj')))
    assert len(body_mesh_fnames)

    for body_mesh_fname in body_mesh_fnames:
        png_out_fname = os.path.join(cfg.dirs.png_out_dir, os.path.basename(body_mesh_fname).replace('.obj', '.png'))
        if os.path.exists(png_out_fname):
            # logger.debug(f'already exists: {png_out_fname}')
            continue

        marker_mesh_fname = body_mesh_fname.replace('/body_mesh/', '/marker_mesh/')
        marker_mesh_fname = marker_mesh_fname.replace('.obj', '.ply')

        if cfg.render.show_markers: assert osp.exists(marker_mesh_fname)

        run_blender_once(cfg, body_mesh_fname, marker_mesh_fname, png_out_fname)

        if cfg.render.render_only_one_image: break

    # if os.path.exists(ps.output_mp4path): return
    if not cfg.render.render_only_one_image:
        if len(glob.glob(os.path.join(cfg.dirs.png_out_dir, '*.png'))) == 0:
            logger.error(f'No images were present at {cfg.dirs.png_out_dir}')
            return
        png_path_pattern = os.path.join(cfg.dirs.png_out_dir, '%*.png')
        pngs2mp4(png_path_pattern, cfg.dirs.mp4_out_fname, fps=cfg.render.video_fps)

        # pngs = sorted(glob.glob(os.path.join(ps.png_outdir, '*.png')), key=os.path.getmtime)
        # pngs2gif(pngs, ps.output_mp4path.replace('.mp4', '.gif'))

        # shutil.rmtree(ps.png_outdir)
        # shutil.rmtree(ps.mesh_dir)
