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
import os.path as osp
from glob import glob

from loguru import logger

from moshpp.mosh_head import MoSh

# amass_pkl_dir = '/is/cluster/scratch/nghorbani/amass/mosh_results/20210726/amass_neutral'
# amass_npz_base_dir = '/is/cluster/scratch/nghorbani/amass/mosh_results/20210726/amass_neutral'
# mosh_stageii_pkl_fnames = glob(osp.join(amass_pkl_dir, '*/*/*stageii.pkl'))
import pickle
from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

amass_pkl_dir = '/ps/project/soma/support_files/release_soma/SOMA_dataset/renamed_subjects/mosh_results'
amass_npz_base_dir = '/ps/project/soma/support_files/release_soma/SOMA_dataset/renamed_subjects/mosh_results_npz'

# amass_pkl_dir = '/ps/project/soma/training_experiments/V48/V48_02_Mixamo/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet'
# amass_npz_base_dir = '/ps/project/soma/training_experiments/V48/V48_02_Mixamo/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet_npz'

# amass_pkl_dir = '/ps/project/soma/training_experiments/V48/V48_02_DanceDB/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet'
# amass_npz_base_dir = '/ps/project/soma/training_experiments/V48/V48_02_DanceDB/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet_npz'

# amass_pkl_dir = '/ps/project/soma/training_experiments/V48/V48_02_CMUII/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet'
# amass_npz_base_dir = '/ps/project/soma/training_experiments/V48/V48_02_CMUII/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet_npz'


mosh_stageii_pkl_fnames = glob(osp.join(amass_pkl_dir, '*/*/*stageii.pkl'))

for mosh_stageii_pkl_fname in sorted(mosh_stageii_pkl_fnames):
    ds_name, subject_name, pkl_basename = mosh_stageii_pkl_fname.split('/')[-3:]

    stageii_npz_fname = osp.join(amass_npz_base_dir, ds_name, subject_name, pkl_basename.replace('.pkl', '.npz'))

    if osp.exists(stageii_npz_fname):
        continue

    stageii_pkl_data = pickle.load(open(mosh_stageii_pkl_fname, 'rb'))

    # stageii_pkl_data['stageii_debug_details']['cfg']['surface_model']['gender'] =  f"{stageii_pkl_data['stageii_debug_details']['cfg']['surface_model']['gender']}"
    # pickle.dump(stageii_pkl_data, open(mosh_stageii_pkl_fname, 'wb'))
    # try:
    MoSh.load_as_amass_npz(mosh_stageii_pkl_fname,
                           stageii_npz_fname=stageii_npz_fname,
                           include_markers=True,
                           )
    # except Exception as e:
    #     logger.error(mosh_stageii_pkl_fname)
    #     os.remove(mosh_stageii_pkl_fname)

# mosh_npz_fnames = glob(osp.join(amass_pkl_dir, '*/*/*.npz'))
# for mosh_npz_fname in mosh_npz_fnames:
#     os.remove(mosh_npz_fname)
