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

import fnmatch
import os.path as osp
import shutil
from glob import glob

import numpy as np
from human_body_prior.tools.omni_tools import flatten_list, makepath


def pick_bmlrub_ghorbani(outdir, amass_dir, fname_filter):
    mosh_stageii_npz_fnames = glob(osp.join(amass_dir, 'BMLrub', '*/*_stageii.npz'))

    for mosh_stageii_npz_fname in mosh_stageii_npz_fnames:
        subject_name, npz_basename = mosh_stageii_npz_fname.split('/')[-2:]

        mosh_stagei_npz_fname = glob(osp.join(osp.dirname(mosh_stageii_npz_fname), '*_stagei.npz'))[0]
        mosh_stagei_pkl_fname = mosh_stagei_npz_fname.replace('.npz', '.pkl')
        mosh_stageii_pkl_fname = mosh_stageii_npz_fname.replace('.npz', '.pkl')

        assert osp.exists(mosh_stagei_npz_fname)
        assert osp.exists(mosh_stagei_pkl_fname)
        assert osp.exists(mosh_stageii_pkl_fname)

        if not np.any([fnmatch.fnmatch(mosh_stageii_npz_fname, a) for a in fname_filter]):
            continue

        dst_mosh_stageii_npz_fname = osp.join(outdir, subject_name, osp.basename(mosh_stageii_npz_fname))
        if not osp.exists(dst_mosh_stageii_npz_fname):
            shutil.copy2(mosh_stageii_npz_fname, makepath(dst_mosh_stageii_npz_fname, isfile=True))

        dst_mosh_stagei_npz_fname = osp.join(outdir, subject_name, osp.basename(mosh_stagei_npz_fname))
        if not osp.exists(dst_mosh_stagei_npz_fname):
            shutil.copy2(mosh_stagei_npz_fname, makepath(dst_mosh_stagei_npz_fname, isfile=True))

        dst_mosh_stagei_pkl_fname = osp.join(outdir, subject_name, osp.basename(mosh_stagei_pkl_fname))
        if not osp.exists(dst_mosh_stagei_pkl_fname):
            shutil.copy2(mosh_stagei_pkl_fname, makepath(dst_mosh_stagei_pkl_fname, isfile=True))

        dst_mosh_stageii_pkl_fname = osp.join(outdir, subject_name, osp.basename(mosh_stageii_pkl_fname))
        if not osp.exists(dst_mosh_stageii_pkl_fname):
            shutil.copy2(mosh_stageii_pkl_fname, makepath(dst_mosh_stageii_pkl_fname, isfile=True))


def main():
    amass_dir = '/ps/project/soma/support_files/release_soma/smplx/amass_neutral'
    outdir_train = osp.join(amass_dir, 'BMLrub_train_ghorbani_permutation_2019')
    train_fname_filter = flatten_list(
        [[f'*{subject_id:03d}/*{action_name}*' for subject_id in range(1, 68)] for action_name in
         ['sit', 'jump', 'walk', 'jog']])
    pick_bmlrub_ghorbani(outdir_train, amass_dir, train_fname_filter)

    outdir_vald = osp.join(amass_dir, 'BMLrub_vald_ghorbani_permutation_2019')
    vald_fname_filter = flatten_list(
        [[f'*{subject_id:03d}/*{action_name}*' for subject_id in range(92, 115)] for action_name in
         ['sit', 'jump', 'walk', 'jog']])
    pick_bmlrub_ghorbani(outdir_vald, amass_dir, vald_fname_filter)


if __name__ == '__main__':
    main()
