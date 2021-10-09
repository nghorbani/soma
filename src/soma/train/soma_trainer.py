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
from collections import OrderedDict
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import get_support_data_dir
from human_body_prior.tools.omni_tools import makepath
from human_body_prior.tools.omni_tools import trainable_params_count
from loguru import logger
from notifiers.logging import NotificationHandler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module

from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.labels_map import general_labels_map
from soma.models.soma_model import SOMA
from soma.tools.eval_tools import find_corresponding_labels, compute_labeling_metrics
from soma.tools.soma_processor import SOMAMoCapPointCloudLabeler

pd.set_option('display.max_rows', 500)


def create_soma_data_id(num_occ_max, num_ghost_max, limit_real_data, limit_synt_data):
    return f'OC_{num_occ_max:02d}_G_{num_ghost_max:02d}_real_{int(limit_real_data * 100):03d}_synt_{int(limit_synt_data * 100):03d}'


class SOMATrainer(LightningModule):
    """

    """

    def __init__(self, cfg: DictConfig):
        super(SOMATrainer, self).__init__()

        self.soma_expr_id = cfg.soma.expr_id
        self.soma_data_id = cfg.soma.data_id

        _support_data_dir = get_support_data_dir(__file__)

        # telegram_cfg = OmegaConf.load(osp.join(_support_data_dir, 'conf/telegram.yaml'))
        # telegram_handler = NotificationHandler("telegram", defaults={'token': telegram_cfg.token,
        #                                                              'chat_id': telegram_cfg.chat_id})

        log_format = f"{cfg.soma.expr_id} -- {cfg.soma.data_id} -- {{message}}"
        # logger.add(telegram_handler, format=log_format, level="SUCCESS")

        logger.info(f'superset_fname: {cfg.data_parms.marker_dataset.superset_fname}')

        superset_meta = marker_layout_load(cfg.data_parms.marker_dataset.superset_fname, labels_map=general_labels_map)

        self.body_parts = superset_meta['marker_type_mask'].keys()

        # breakpoint()
        # self.body_parts = ['body']

        self.use_multiple_body_parts = len(self.body_parts) > 1

        self.work_dir = cfg.dirs.work_dir

        self.enable_sinkhorn = cfg.model_parms.labeler.enable_sinkhorn
        self.train_start_time = dt.now().replace(microsecond=0)

        self.soma_model = SOMA(cfg)

        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.cfg_fname = osp.join(self.work_dir, f'{self.soma_expr_id}_{self.soma_data_id}.yaml')
        OmegaConf.save(config=self.cfg, f=self.cfg_fname)

        self.renderer = None

    def forward(self, points):
        return self.soma_model(points)

    @rank_zero_only
    def on_train_start(self):
        # if self.global_rank != 0: return
        # shutil.copy2(__file__, self.work_dir)

        ######## make a backup of soma
        git_repo_dir = os.path.abspath(__file__).split('/')
        git_repo_dir = '/'.join(git_repo_dir[:git_repo_dir.index('soma') + 1])
        start_time = dt.strftime(self.train_start_time, '%Y_%m_%d_%H_%M_%S')
        archive_path = makepath(self.work_dir, 'code', f'soma_{start_time}.tar.gz', isfile=True)
        os.system(f"cd {git_repo_dir} && git ls-files -z | xargs -0 tar -czf {archive_path}")
        ########
        logger.info(f'Created a git archive backup at {archive_path}')

    def configure_optimizers(self):

        lr_scheduler_class = getattr(lr_sched_module, self.cfg.train_parms.lr_scheduler.type)
        schedulers = []
        optimizers = []

        gen_params = [a[1] for a in self.soma_model.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim_module, self.cfg.train_parms.gen_optimizer.type)
        gen_optimizer = gen_optimizer_class(gen_params, **self.cfg.train_parms.gen_optimizer.args)
        optimizers.append(gen_optimizer)
        gen_lr_scheduler = lr_scheduler_class(gen_optimizer, **self.cfg.train_parms.lr_scheduler.args)
        schedulers.append({
            'scheduler': gen_lr_scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        })

        logger.info(f'Total trainable gen_params: {trainable_params_count(gen_params) * 1e-6:2.4f} M.')

        return optimizers, schedulers

    def _compute_labeling_loss(self, aug_asmat_rec, aug_asmat_gt, aug_asmat_weights):
        if self.enable_sinkhorn:
            nominator = aug_asmat_gt * aug_asmat_rec  # .view(aug_asmat_gt.shape)
            weighted_nominator = nominator * aug_asmat_weights
            nll = (weighted_nominator).sum([1, 2], keepdim=True) / aug_asmat_gt.sum([1, 2], keepdim=True)
            return - nll.sum()  # .sum(-1).mean()
        else:
            nominator = aug_asmat_gt[:, :-1] * aug_asmat_rec  # .view(aug_asmat_gt[:,:-1].shape)
            weighted_nominator = nominator * aug_asmat_weights[:, :-1]
            nll = (weighted_nominator).sum([1, 2], keepdim=True) / aug_asmat_gt[:, :-1].sum([1, 2], keepdim=True)
            return - nll.sum()  # .sum(-1).mean()

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        cfg = self.cfg

        wts = cfg.train_parms.loss_weights

        drec = self.forward(batch['points'])

        if optimizer_idx == 0 or optimizer_idx is None:  # train generator

            label_assignment_loss = self._compute_labeling_loss(
                drec['aug_asmat'] if self.enable_sinkhorn else drec['label_confidence'],
                batch['aug_asmat'], batch['aug_asmat_weights'])

            # self.log('train_label_loss', label_assignment_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            train_loss = wts.labeling * label_assignment_loss

            if self.use_multiple_body_parts:
                raise NotImplementedError('This functionality is not released for current SOMA.')

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx, mode='vald'):
        batch_size, num_points, _ = batch['points'].shape
        superset_labels = self.trainer.datamodule.superset_labels

        drec = self.forward(batch['points'])

        loss = self._compute_labeling_loss(
            drec['aug_asmat'] if self.enable_sinkhorn else drec['label_confidence'],
            batch['aug_asmat'], batch['aug_asmat_weights'])

        # node_label_conf, node_label_id = drec['label_confidence'].max(-1)

        # markers_rec = []
        # labels_rec = []
        # if self.current_epoch > 0:
        #     breakpoint()
        res = SOMAMoCapPointCloudLabeler.create_mocap(points=c2c(batch['points']),
                                                      points_label_id=c2c(drec['label_confidence'].argmax(-1)),
                                                      superset_labels=np.array(superset_labels),
                                                      keep_nan_points=True, remove_zero_trajectories=False)
        # markers_rec.append(res['markers'])
        # labels_rec.append(res['label_ids_perframe'])

        result = {'val_loss': c2c(loss).reshape(1),  # }
                  'markers_gt': c2c(batch['points']),  # .view(-1, num_points, 3)),
                  'markers_rec': res['markers'],
                  'labels_rec': superset_labels[c2c(res['label_ids_perframe'])],
                  # 'label_conf': np.concatenate(labels_soma_conf),
                  'labels_gt': superset_labels[c2c(batch['label_ids'])]}

        if self.use_multiple_body_parts:
            raise NotImplementedError('This functionality is not released for current SOMA.')

        return result

    def validation_epoch_end(self, outputs):
        # if self.global_rank != 0: return
        superset_labels = self.trainer.datamodule.superset_labels

        data = {}
        for one in outputs:
            for k, v in one.items():
                if k == 'progress_bar': continue
                if not k in data: data[k] = []
                data[k].append(v)
        # data = {k: np.concatenate(v) for k, v in data.items()}

        res_perbatch = {'f1': [], 'acc': []}
        for batch_id in range(len(data['markers_gt'])):
            res_aligned = find_corresponding_labels(
                markers_gt=data['markers_gt'][batch_id],
                labels_gt=data['labels_gt'][batch_id],
                markers_rec=data['markers_rec'][batch_id],
                labels_rec=data['labels_rec'][batch_id],
                flatten_output=True,
            )

            res = compute_labeling_metrics(res_aligned['labels_gt'], res_aligned['labels_rec'], create_excel_dfs=False)
            for k in res_perbatch:
                res_perbatch[k].append(res[k])

        metrics = OrderedDict({'val_loss': np.nanmean(np.concatenate([v['val_loss'] for v in outputs])),
                               'f1': np.mean(res_perbatch['f1']),
                               'acc': np.mean(res_perbatch['acc']),
                               })

        if self.global_rank == 0:

            logger.success(
                f'Epoch {self.current_epoch}: {", ".join(f"{k}:{v:.2f}" for k, v in metrics.items())}')
            logger.info(
                f'lr is {["{:.2e}".format(pg["lr"]) for opt in self.trainer.optimizers for pg in opt.param_groups]}')

        metrics = {k if k.startswith('val_') else f'val_{k}': torch.tensor(v, device=self.device) for k, v in
                   metrics.items()}

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @rank_zero_only
    def on_train_end(self):

        self.train_endtime = dt.now().replace(microsecond=0)
        endtime = dt.strftime(self.train_endtime, '%Y_%m_%d_%H_%M_%S')
        elapsedtime = self.train_endtime - self.train_start_time

        best_model_basename = self.trainer.checkpoint_callback.best_model_path
        logger.success(f'best_model_fname: {best_model_basename}')
        OmegaConf.save(config=self.cfg, f=self.cfg_fname)

        logger.success(f'Epoch {self.current_epoch} - Finished training at {endtime} after {elapsedtime}')

    @staticmethod
    def prepare_cfg(**kwargs) -> DictConfig:

        if not OmegaConf.has_resolver('resolve_soma_data_id'):
            OmegaConf.register_new_resolver('resolve_soma_data_id',
                                            lambda num_occ_max, num_ghost_max, limit_real_data, limit_synt_data:
                                            create_soma_data_id(num_occ_max, num_ghost_max, limit_real_data,
                                                                limit_synt_data))

        app_support_dir = get_support_data_dir(__file__)
        base_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/soma_train_conf.yaml'))

        override_cfg_dotlist = [f'{k}={v}' for k, v in kwargs.items()]
        override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

        return OmegaConf.merge(base_cfg, override_cfg)


def train_soma_once(job_args):
    """
    This function must be self sufficient with imports to be able to run on the cluster
    :param job_args:
    :return:
    """
    from soma.train.soma_trainer import SOMATrainer
    from soma.train.soma_data_module import SOMADATAModule

    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelCheckpoint

    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.loggers import CSVLogger

    from pytorch_lightning.plugins import DDPPlugin

    import glob
    import os
    import os.path as osp
    import sys

    import torch
    from human_body_prior.tools.omni_tools import makepath
    from loguru import logger

    cfg = SOMATrainer.prepare_cfg(**job_args)

    if cfg.trainer.deterministic:
        pl.seed_everything(cfg.trainer.rnd_seed, workers=True)

    log_format = f"{{module}}:{{function}}:{{line}} -- {cfg.soma.expr_id} -- {cfg.soma.data_id} -- {{message}}"

    logger.remove()

    logger.add(cfg.dirs.log_fname, format=log_format, enqueue=True)
    logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

    soma_dm = SOMADATAModule(cfg)
    soma_dm.prepare_data()
    soma_dm.setup(stage='fit')

    model = SOMATrainer(cfg)

    makepath(cfg.dirs.log_dir, 'tensorboard')
    makepath(cfg.dirs.log_dir, 'csv')
    tboard_logger = TensorBoardLogger(cfg.dirs.log_dir, name='tensorboard')
    csv_logger = CSVLogger(cfg.dirs.log_dir, name='csv', version=None, prefix='')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    snapshots_dir = osp.join(model.work_dir, 'snapshots')
    checkpoint_callback = ModelCheckpoint(
        dirpath=makepath(snapshots_dir, isfile=True),
        filename="%s_{epoch:02d}_{val_f1:.2f}_{val_acc:.2f}" % model.cfg.soma.expr_id,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    early_stop_callback = EarlyStopping(**model.cfg.train_parms.early_stopping)

    resume_checkpoint_fname = cfg.trainer.resume_checkpoint_fname
    if cfg.trainer.resume_training_if_possible and not cfg.trainer.fast_dev_run:
        if not resume_checkpoint_fname:
            available_ckpts = sorted(glob.glob(osp.join(snapshots_dir, '*.ckpt')), key=os.path.getmtime)
            if len(available_ckpts) > 0:
                resume_checkpoint_fname = available_ckpts[-1]
    if resume_checkpoint_fname:
        logger.info(f'Resuming the training from {resume_checkpoint_fname}')

    if cfg.trainer.finetune_checkpoint_fname:  # only reuse weights and not the learning rates
        state_dict = torch.load(cfg.trainer.finetune_checkpoint_fname)['state_dict']
        model.load_state_dict(state_dict, strict=True)
        # Todo fix the issues so that we can set the strict to true. The body model uses unnecessary registered buffers
        logger.info(f'Loaded finetuning weights from {cfg.trainer.finetune_checkpoint_fname}')

    trainer = pl.Trainer(gpus=1 if cfg.trainer.fast_dev_run else cfg.trainer.num_gpus,
                         weights_summary=cfg.trainer.weights_summary,
                         distributed_backend=None if cfg.trainer.fast_dev_run else cfg.trainer.distributed_backend,
                         profiler=cfg.trainer.profiler,
                         plugins=None if cfg.trainer.fast_dev_run else [DDPPlugin(find_unused_parameters=False)],
                         fast_dev_run=cfg.trainer.fast_dev_run,
                         limit_train_batches=cfg.trainer.limit_train_batches,
                         limit_val_batches=cfg.trainer.limit_val_batches,
                         num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,  # run full validation run
                         callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],
                         max_epochs=cfg.trainer.max_epochs,
                         logger=[tboard_logger, csv_logger],
                         resume_from_checkpoint=resume_checkpoint_fname,
                         deterministic=cfg.trainer.deterministic,
                         )

    trainer.fit(model, soma_dm)
