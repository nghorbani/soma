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

import numpy as np
import torch


# def make_ghost_points(markers, num_ghost_max, use_exact_num_ghost=False):
#     '''
#
#     Args:
#         markers: Txnum_pointsx3
#         num_ghost_max:
#
#     Returns:
#
#     '''
#     if num_ghost_max == 0: return None
#
#     T, num_markers = markers.shape[:-1]
#
#     mrk_median, mrk_std = torch.median(markers, 1).values, markers.std(1)
#     mrk_std_batched = torch.eye(3).reshape((1, 3, 3)).repeat(T, 1, 1) * mrk_std.reshape((-1, 1, 3))
#     ghost_mrk_generator = torch.distributions.MultivariateNormal(mrk_median, scale_tril=mrk_std_batched)
#
#     ghost_markers = torch.cat([ghost_mrk_generator.sample()[:, None] for _ in range(num_ghost_max)], dim=1)
#
#     if use_exact_num_ghost:
#
#         # Number of ghost markers should vary across frames. to do this we append a dummy marker
#         # and sample from a random generator that produces X times more than the number of ghost markers
#         # any number in the mask bigger than the actual number of ghosts is set to the dummy marker
#         ghost_markers = torch.cat([ghost_markers, torch.zeros(T, 1, 3)], dim=1)
#         ghost_reject_mask = np.random.randint(3 * num_ghost_max, size=(T, num_ghost_max))
#         ghost_reject_mask[ghost_reject_mask >= num_ghost_max] = ghost_markers.shape[1] - 1
#         ghost_reject_mask = np.repeat(ghost_reject_mask[:, :, None], 3, axis=-1)
#         np.put_along_axis(ghost_markers, ghost_reject_mask, 0, axis=1)
#         ghost_markers = ghost_markers[:, :-1]
#
#
#     return ghost_markers

def make_ghost_points(markers, num_ghost_max, ghost_distribution='spherical_gaussian', use_upto_num_ghost=False):
    # todo: do you really need use_exact_num_ghost.
    # todo: is use_exact_num_ghost doing what it is intended for?

    assert ghost_distribution in ['spherical_gaussian', 'uniform', 'skewed_gaussian']

    if num_ghost_max == 0: return None

    time_length, num_markers = markers.shape[:-1]

    mrk_median, mrk_std = torch.median(markers, 1).values, markers.std(1)
    mrk_std_batched = torch.eye(3).reshape((1, 3, 3)).repeat(time_length, 1, 1) * mrk_std.reshape((-1, 1, 3))

    if ghost_distribution == 'spherical_gaussian':
        ghost_mrk_generator = torch.distributions.MultivariateNormal(mrk_median, scale_tril=mrk_std_batched)

        ghost_markers = torch.cat([ghost_mrk_generator.sample()[:, None] for _ in range(num_ghost_max)], dim=1)

    elif ghost_distribution == 'uniform':
        assert time_length == 1
        uniform_dist = torch.distributions.uniform.Uniform(low=-2, high=2)
        ghost_markers = \
            torch.stack([torch.stack([uniform_dist.sample() for _ in range(3)]) for _ in range(num_ghost_max)], dim=0)[
                None]

    elif ghost_distribution == 'skewed_gaussian':
        assert time_length == 1

        from scipy.stats import random_correlation
        random_eigens = np.random.uniform(size=3)
        random_eigens = (random_eigens / random_eigens.sum()) * 3
        random_cov = random_correlation.rvs(random_eigens)
        # random_cov = random_cov.dot(random_cov.time_length)
        random_mean = np.random.uniform(low=-2, high=2, size=3)

        ghost_mrk_generator = torch.distributions.MultivariateNormal(mrk_median.new(random_mean[None]),
                                                                     covariance_matrix=mrk_std_batched.new(
                                                                         random_cov[None]))

        ghost_markers = torch.cat([ghost_mrk_generator.sample()[:, None] for _ in range(num_ghost_max)], dim=1)

    else:
        raise NotImplementedError

    if use_upto_num_ghost:
        # Number of ghost markers should vary across frames. to do this we append a dummy marker
        # and sample from a random generator that produces X times more than the number of ghost markers
        # any number in the mask bigger than the actual number of ghosts is set to the dummy marker
        ghost_markers = torch.cat([ghost_markers, torch.zeros(time_length, 1, 3)], dim=1)
        ghost_reject_mask = np.random.randint(3 * num_ghost_max, size=(time_length, num_ghost_max))
        ghost_reject_mask[ghost_reject_mask >= num_ghost_max] = ghost_markers.shape[1] - 1
        ghost_reject_mask = np.repeat(ghost_reject_mask[:, :, None], 3, axis=-1)
        np.put_along_axis(ghost_markers, ghost_reject_mask, 0, axis=1)
        ghost_markers = ghost_markers[:, :-1]

    return ghost_markers


# def occlude_points(markers, num_occ_max, use_exact_num_oc=False):
#     '''
#
#     Args:
#         markers: Txnum_pointsx3
#         num_occ_max:
#
#     Returns:
#
#     '''
#     T, num_markers = markers.shape[:-1]
#
#     markers = torch.cat([markers, torch.zeros(T, 1, 3)], dim=1)
#     if use_exact_num_oc:
#         occ_mask = []
#         for t in range(T):
#             occ_mask.append(np.random.choice(num_markers, size=num_occ_max, replace=False))
#         occ_mask = np.stack(occ_mask)
#     else:
#         occ_mask = np.random.randint(3 * num_markers, size=(T, num_occ_max))
#         occ_mask[occ_mask >= num_markers] = markers.shape[1] - 1
#
#     occ_mask = np.repeat(occ_mask[:, :, None], 3, axis=-1)#so that all x,y,z channels are flattened
#
#
#     np.put_along_axis(markers, occ_mask, 0, axis=1)
#     markers = markers[:, :-1]
#     return markers

def occlude_markers(markers, num_occ):
    '''

    Args:
        markers: num_markers x 3
        num_occ:

    Returns:

    '''
    if num_occ == 0: return markers
    num_markers = markers.shape[0]

    occ_mask = np.random.choice(num_markers, size=num_occ, replace=False)

    occ_mask = np.repeat(occ_mask[:, None], 3, axis=-1)  # to effect xyz

    np.put_along_axis(markers, occ_mask, 0, axis=0)
    return markers


def break_trajectories(markers, label_ids, nan_class_id, num_btraj_max):
    '''

    Args:
        markers: Txnum_pointsx3
        label_ids:
        nan_class_id:
        num_btraj_max:

    Returns:

    '''
    T, num_markers = markers.shape[:-1]

    selection_ids = list(range(label_ids.shape[1]))
    add_count = label_ids.shape[1] - 1
    for t in sorted(np.random.choice(np.arange(2, T - 1), np.minimum(num_btraj_max, T), replace=False)):
        # at each t in t_list a trajectory will be cut and placed at a new index with the same label
        # t should monotonically grow
        # print(t)
        # if np.random.rand() > 0.5: continue
        i = np.random.choice(len(selection_ids))
        mrk_id = selection_ids.pop(i)  # this id will be cut
        while label_ids[t, mrk_id] == nan_class_id:
            i = np.random.choice(len(selection_ids))
            mrk_id = selection_ids.pop(i)

        mrk_data = torch.cat([torch.zeros_like(markers[:t, mrk_id]), markers[t:, mrk_id]])
        label_data = torch.cat([torch.tensor([nan_class_id] * t), label_ids[t:, mrk_id]])

        markers[t:, mrk_id] = 0.0
        label_ids[t:, mrk_id] = nan_class_id
        markers = torch.cat([markers, mrk_data[:, None]], axis=1)
        label_ids = torch.cat([label_ids, label_data[:, None]], axis=1)
        add_count = add_count + 1
        selection_ids.append(add_count)  # add the pasted id so that it can be cut again by chance
        # mrks_labels2.append(mrks_labels2[mrk_id])
        # print('cut mrk_id %d, label %s at time t=%d'%(mrk_id, label_ids[t-1,mrk_id], t))
        # continue
    return markers, label_ids
