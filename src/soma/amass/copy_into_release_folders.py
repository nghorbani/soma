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
# @inproceedings{GhorbaniBlack:ICCV:2021,
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
import os.path as osp
import shutil
from collections import OrderedDict
from glob import glob

import numpy as np
from human_body_prior.tools.omni_tools import makepath
from loguru import logger
from tqdm import tqdm

# only_datasets = sorted(['HumanEva', 'ACCAD', 'PosePrior'])
# only_datasets = sorted(['HumanEva', 'ACCAD', 'TotalCapture', 'CMU', 'Transitions', 'PosePrior'])
only_datasets = None
release_base_dir = '/ps/project/datasets/AMASS/smplx/neutral'
amass_npz_base_dir = '/is/cluster/scratch/nghorbani/amass/mosh_results/20210726/amass_neutral'
amass_mp4_base_dir = '/is/cluster/scratch/nghorbani/amass/mp4_renders/20210726/amass_neutral'
license_fname = '/ps/project/datasets/AMASS/LICENSE.txt'
assert osp.exists(license_fname)
mosh_npz_fnames = glob(osp.join(amass_npz_base_dir, '*/*/*.npz'))

amass_stats = OrderedDict()
for npz_fname in tqdm(sorted(mosh_npz_fnames)):
    ds_name, subject_name, npz_basename = npz_fname.split('/')[-3:]
    if (only_datasets is not None) and (ds_name not in only_datasets): continue

    if npz_basename.endswith('_stageii.npz'):
        render_fname = osp.join(amass_mp4_base_dir, ds_name, subject_name, npz_basename.replace('_stageii.npz', '.mp4'))

        new_npz_fname = osp.join(release_base_dir, 'mosh_results', ds_name, subject_name, npz_basename)
        new_render_fname = osp.join(release_base_dir, 'mp4_renders', ds_name, subject_name,
                                    npz_basename.replace('_stageii.npz', '.mp4'))
        new_license_mosh_fname = osp.join(release_base_dir, 'mosh_results', ds_name, 'LICENSE.txt')
        new_license_mp4_fname = osp.join(release_base_dir, 'mp4_renders', ds_name, 'LICENSE.txt')

        if not osp.exists(new_npz_fname):
            mosh_data = np.load(npz_fname)  # see if it is a valid npz
            shutil.copy2(npz_fname, makepath(new_npz_fname, isfile=True))

        if not osp.exists(new_render_fname):
            if osp.exists(render_fname):
                shutil.copy2(render_fname, makepath(new_render_fname, isfile=True))
            else:
                logger.error(f'render_fname does not exist: {render_fname}')

        if not osp.exists(new_license_mosh_fname):
            shutil.copy2(license_fname, makepath(new_license_mosh_fname, isfile=True))

        if not osp.exists(new_license_mp4_fname):
            shutil.copy2(license_fname, makepath(new_license_mp4_fname, isfile=True))

    else:  # stagei
        new_npz_fname = osp.join(release_base_dir, 'mosh_results', ds_name, subject_name, npz_basename)
        if not osp.exists(new_npz_fname):
            shutil.copy2(npz_fname, makepath(new_npz_fname, isfile=True))


def compress_folder(directory):
    dir_basename = osp.basename(directory)
    root_dir = osp.dirname(directory)
    if not osp.exists(f'{directory}.tar.bz2'):
        os.system(f'cd {root_dir}; tar cjvf {dir_basename}.tar.bz2 {dir_basename}')


for directory in glob(osp.join(release_base_dir, 'mosh_results/*')):
    if osp.isdir(directory):
        compress_folder(directory)

for directory in glob(osp.join(release_base_dir, 'mp4_renders/*')):
    if osp.isdir(directory):
        compress_folder(directory)
