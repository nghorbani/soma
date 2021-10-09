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
from setuptools import setup, find_packages
from glob import glob

setup(name='soma',
      version='5.0.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      data_files=[
          ('soma/support_data', glob('support_data/*.*')),
          ('soma/support_data/conf/parallel_conf', glob('support_data/conf/parallel_conf/*.*')),
          ('soma/support_data/github_files', glob('support_data/github_files/*.*')),
          ('soma/support_data/tests', glob('support_data/tests/*.*')),
                  ('soma/support_data/conf', glob('support_data/conf/*.*'))
      ],

      author='Nima Ghorbani',
      author_email='nghorbani@tue.mpg.de',
      maintainer='Nima Ghorbani',
      maintainer_email='nghorbani@tue.mpg.de',
      url='https://github.com/nghorbani/soma',
      description='Solving Optical Marker-Based Motion Capture Automatically',
      license='See LICENSE.txt',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=[],
      dependency_links=[],
      classifiers=[
          "Intended Audience :: Research",
          "Natural Language :: English",
          "Operating System :: POSIX",
          "Operating System :: POSIX :: BSD",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7", ],
      )
