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
from glob import glob
from os import path as osp
import os.path as osp
amass_mocap_base_dir = '/ps/project/amass/MOCAP'

amass_datasets = {
    'ACCAD': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'ACCAD', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        }
    },
    'BMLHandball': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'BMLhandball/pkl', '*/*.pkl')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        },
    },
    'BMLmovi': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'BMLmovi/2019_09_24', '*/*.pkl')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_toes': True,
            'mocap.rotate': [0, 0, -90],
        }
    },
    'BMLrub': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'BMLrub/1999_rub/pkl', '*/*.pkl')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        },
        'subject_specific_settings': {
            'rub002': {
                'mocap.exclude_markers': ['LFHD']
            }
        },
    },
    'CMU': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'CMU/c3d/subjects', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.wrist_markers_on_stick': True
        }
    },
    # 'CMUII': {
    #     'mocap_fnames': glob(osp.join(amass_mocap_base_dir, '/CMU_II/SOMA_V48_02/CMUII_unlabeled_mpc', '*/*.c3d')),
    #     'mosh_cfg_override': {
    #         'mocap.unit': 'mm',
    #         'surface_model.gender': '${resolve_gender:${mocap.fname},neutral}'
    #     },
    # },
    'CMU_MS': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'CMU/c3d/multisubject', '*/*.npz')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_dynamics': False
        }
    },
    'CNRS': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'KIT_Whole_Body/CNRS', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        }
    },
    'DFaust': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'DFAUST', '*/*.npz')),
        'mosh_cfg_override': {
            'mocap.unit': 'm',
            'moshpp.optimize_toes': True,
        },
        'render_cfg_override': {
            'render.video_fps': 10,
            'mesh.ds_rate': 2,
        }
    },
    'DanceDB': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'DanceDB/SOMA_V48_02/DanceDB_c3d_120hz', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        },
    },
    'EKUT': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'KIT_Whole_Body/EKUT', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        }
    },
    'Eyes_Japan_Dataset': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'Eyes_Japan_Dataset', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.wrist_markers_on_stick': True,
            'moshpp.optimize_dynamics': False
        }
    },
    # 'GRAB': {
    #     'mocap_fnames': glob(
    #         osp.join(amass_mocap_base_dir, 'PS_MoCaps/GRAB/GRAB_manual_labeled_gap_filled', '*/*.c3d')),
    #     'mosh_cfg_override': {
    #         'mocap.unit': 'mm',
    #         'moshpp.optimize_toes': True,
    #         'moshpp.optimize_fingers': True,
    #         'moshpp.optimize_betas': False,
    #         'opt_settings.weights_type': 'smplx_grab_vtemplate',
    #         'moshpp.stagei_frame_picker.least_avail_markers': 1.0,
    #
    #         'moshpp.separate_types': ['body', 'finger', 'face'],
    #         'subject_specific_settings': {subject_name:
    #                                           {'moshpp.v_template_fname':f'/ps/project/amass/MOCAP/PS_MoCaps/GRAB/subject_meshes/{subject_name}.ply'}
    #                                                     for subject_name in [f's{d}' for d in range(1,11)]},
    #         'dirs.marker_layout_fname': '/ps/project/amass/MOCAP/PS_MoCaps/GRAB/marker_layout/s4/apple_eat_1.c3d',
    #     }
    # },
    'HDM05': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'MPI_HDM05', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_dynamics': False
        }
    },
    'HUMAN4D': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'HUMAN4D/mocap_pkls', '*/*.pkl')),
        'mosh_cfg_override': {
            'mocap.unit': 'm',
            'moshpp.optimize_dynamics': False,
            'moshpp.optimize_toes': False,
        }
    },
    'HumanEva': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'HumanEva/mocap_for_mosh', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.wrist_markers_on_stick': True,
            'moshpp.optimize_dynamics': False
        }
    },
    'KIT': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'KIT_Whole_Body/KIT', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_dynamics': False
        }
    },
    'LAFAN1': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'LAFAN1/mocap_pkls', '*/*.pkl')),
        'mosh_cfg_override': {
            'mocap.unit': 'm',
            'moshpp.optimize_dynamics': False
        }
    },
    'SNU': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'SNU/initial_website_scrap', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_dynamics': False,
            'moshpp.optimize_toes': False,
        }
    },

    'MoSh': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'MPI_mosh/c3d/subjects', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.wrist_markers_on_stick': True,
        }
    },
    'PosePrior': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'MPI_Limits/joint-angle-limits/cleaned_data', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'mocap.rotate': [90, 0, 0],
            'moshpp.wrist_markers_on_stick': True,
        },
        # 'subject_specific_settings': {
        #     '03099': {
        #         'moshpp.wrist_markers_on_stick': True
        #     }
        # },
    },
    'SFU': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'SFU', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'mocap.rotate': [90, 0, 0],
        }
    },
    'Rokoko': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'Rokoko/c3d', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'mocap.rotate': [90, 0, 0],
        }
    },
    'SOMA': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'PS_MoCaps/SOMA/SOMA_manual_labeled', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        },
    },
    'SSM': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'SSM_synced/resynced', '*/*.pkl')),
        'persubject_marker_layout': True,
        'render_cfg_override': {
            'mesh.ds_rate': 1,
            'render.video_fps': 60,
        },
        'mosh_cfg_override': {
            'mocap.unit': 'm',
            'moshpp.optimize_toes': True,
            'mocap.rotate': [90, 0, -90],
        },
        'subject_specific_settings': {
            '20160330_03333': {
                'moshpp.wrist_markers_on_stick': True
            }
        },
    },
    'TCDHands': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'TCD_handMocap/c3d_fullmkrs', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_fingers': True,
            'moshpp.optimize_toes': False,
        }
    },
    'TotalCapture': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'TotalCapture/C3D', '*/*.pkl')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'mocap.rotate': [90, 0, 0],
        }
    },
    'Transitions': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'Transitions_mocap/c3d', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'mocap.rotate': [90, 0, 180],
        }
    },
    'WEIZMANN': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'KIT_Whole_Body/WEIZMANN', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        }
    },
}

if __name__ == '__main__':
    for k in sorted(amass_datasets.keys()):
        print(f"'{k}',")
