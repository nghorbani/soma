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

amass_mocap_base_dir = '/ps/project/amass/MOCAP'

amass_datasets = {
    'ACCAD': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'ACCAD', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
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
    'BMLmovi': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'BMLmovi/2019_09_24', '*/*.pkl')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_toes': True,
            'mocap.rotate': [0, 0, -90],
        }
    },
    'CMU': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'CMU/c3d/subjects', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.wrist_markers_on_stick': True
        }
    },
    'CMU_MS': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'CMU/c3d/multisubject', '*/*.npz')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_dynamics': False
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
    'HDM05': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'MPI_HDM05', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.optimize_dynamics': False
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
    'MoSh': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'MPI_mosh/c3d/subjects', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'moshpp.wrist_markers_on_stick': True,
        }
    },
    'SFU': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'SFU', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
            'mocap.rotate': [90, 0, 0],
        }
    },
    # 'GRAB': {
    #     'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'GRAB/new_mocaps', '*/*.c3d')),
    #     'mosh_cfg_override': {
    #         'mocap.unit': 'mm',
    #         'moshpp.optimize_toes': True,
    #         'dirs.marker_layout_fname': '/ps/project/amass/MOCAP/GRAB/smplx_00174.json'
    #     }
    # },
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
    'DFaust': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'DFAUST', '*/*.npz')),
        'mosh_cfg_override': {
            'mocap.unit': 'm',
            'moshpp.optimize_toes': True,
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
    'EKUT': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'KIT_Whole_Body/EKUT', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        }
    },
    'CNRS': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'KIT_Whole_Body/CNRS', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        }
    },
    'WEIZMANN': {
        'mocap_fnames': glob(osp.join(amass_mocap_base_dir, 'KIT_Whole_Body/WEIZMANN', '*/*.c3d')),
        'mosh_cfg_override': {
            'mocap.unit': 'mm',
        }
    },
}
