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
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd

from soma.tools.eval_tools import save_xlsx

# only_datasets = sorted(['HumanEva', 'ACCAD', 'PosePrior'])
# only_datasets = sorted(['HumanEva', 'ACCAD', 'TotalCapture', 'CMU', 'Transitions', 'PosePrior'])
only_datasets = None
# amass_npz_base_dir = '/ps/project/soma/training_experiments/V48/V48_02_DanceDB/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet_npz'
# amass_npz_base_dir = '/ps/project/soma/training_experiments/V48/V48_02_Mixamo/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet_npz'
amass_npz_base_dir = '/ps/project/soma/training_experiments/V48/V48_02_SOMA/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet_npz'

# amass_npz_base_dir = '/is/cluster/scratch/nghorbani/amass/mosh_results/20210726/amass_neutral'

# mosh_stageii_npz_fnames = glob(osp.join(amass_npz_base_dir, '*/*stageii.npz'))
mosh_stageii_npz_fnames = glob(osp.join(amass_npz_base_dir, '*/*/*stageii.npz'))

amass_stats = OrderedDict()
for stageii_npz_fname in sorted(mosh_stageii_npz_fnames):
    ds_name, subject_name, npz_basename = stageii_npz_fname.split('/')[-3:]
    if (only_datasets is not None) and (ds_name not in only_datasets): continue

    mosh_data = np.load(stageii_npz_fname)
    if ds_name not in amass_stats: amass_stats[ds_name] = OrderedDict({'markers': [], 'subjects': [],
                                                                       'motions': 0, 'minutes': 0})

    amass_stats[ds_name]['markers'].append(len(mosh_data['markers_latent']))
    amass_stats[ds_name]['subjects'].append(subject_name)
    amass_stats[ds_name]['motions'] += 1
    amass_stats[ds_name]['minutes'] += mosh_data['mocap_time_length'] / 60.

for ds_name in amass_stats:
    amass_stats[ds_name]['markers'] = np.median(amass_stats[ds_name]['markers'])
    amass_stats[ds_name]['subjects'] = len(np.unique(amass_stats[ds_name]['subjects']))
amass_data_pd = pd.DataFrame(amass_stats).transpose()
xlsx_data = {'amass': amass_data_pd}

save_xlsx(xlsx_data, xlsx_fname=osp.join(amass_npz_base_dir, 'amass_stats.xlsx'))
print(amass_data_pd)
# mosh_npz_fnames = glob(osp.join(amass_pkl_dir, '*/*/*.npz'))
# for mosh_npz_fname in mosh_npz_fnames:
#     os.remove(mosh_npz_fname)
