#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import numpy as np
from utils import logging
from datetime import datetime as dt
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.data_loaders
import utils.data_transforms
import utils.helpers

import core.pipeline_test as pipeline

from models.encoder.encoder_vit_ivdb import Encoder
from models.merger.merger_stm import Merger
from models.decoder.decoder_retr import Decoder

from losses.losses import DiceLoss, CEDiceLoss, FocalLoss
from utils.average_meter import AverageMeter


def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_file_num=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # load data
    taxonomies, test_data_loader, test_file_num = pipeline.load_data(cfg, test_data_loader, test_file_num)

    # Set up networks
    if decoder is None or encoder is None or Merger is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        merger = Merger(cfg)

        encoder, decoder, merger, epoch_idx = \
            pipeline.setup_network(cfg, encoder, decoder, merger)

    # Set up loss functions
    if cfg.TRAIN.LOSS == 1:
        loss_function = torch.nn.BCELoss()
    elif cfg.TRAIN.LOSS == 2:
        loss_function = DiceLoss()
    elif cfg.TRAIN.LOSS == 3:
        loss_function = CEDiceLoss()
    elif cfg.TRAIN.LOSS == 4:
        loss_function = FocalLoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = []
    taxonomies_list = []
    losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    merger.eval()
    
    for_tqdm = tqdm(enumerate(test_data_loader), total=n_samples)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in for_tqdm:
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images).to(torch.cuda.current_device())
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume).to(torch.cuda.current_device())

            # Test the encoder, decoder and merger
            # encoder
            image_features = encoder(rendering_images)

            # merger
            context = merger(image_features)

            # decoder
            generated_volume = decoder(context).squeeze(dim=1)

            generated_volume = generated_volume.clamp_max(1)

            # Loss
            loss = loss_function(generated_volume, ground_truth_volume)

            # Append loss and accuracy to average metrics
            loss = utils.helpers.reduce_value(loss)
            losses.update(loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).unsqueeze(dim=0))
            test_iou.append(torch.cat(sample_iou).unsqueeze(dim=0))
            taxonomies_list.append(torch.tensor(list(taxonomies.keys()).index(taxonomy_id)).unsqueeze(dim=0))

            if True:
                # Print sample loss and IoU
                if (sample_idx + 1) % 50 == 0:
                    for_tqdm.update(50)
                    for_tqdm.set_description('Test[%d/%d] Taxonomy = %s Loss = %.4f' %
                                             (sample_idx + 1, n_samples, taxonomy_id, losses.avg))

                logging.debug('Test[%d/%d] Taxonomy = %s Sample = %s Loss = %.4f IoU = %s' %
                              (sample_idx + 1, n_samples, taxonomy_id, sample_name,
                               loss.item(), ['%.4f' % si for si in sample_iou]))

    test_iou = torch.cat(test_iou, dim=0)
    taxonomies_list = torch.cat(taxonomies_list).to(torch.cuda.current_device())

    test_iou = pipeline.combine_test_iou(test_iou, taxonomies_list, list(taxonomies.keys()), test_file_num)

    torch.cuda.synchronize(torch.device(torch.cuda.current_device()))

    if True:
        # Output testing results
        mean_iou = pipeline.output(cfg, test_iou, taxonomies)

        # Add testing results to TensorBoard
        max_iou = np.max(mean_iou)
        if test_writer is not None:
            test_writer.add_scalar('EpochLoss', losses.avg, epoch_idx)
            test_writer.add_scalar('IoU', max_iou, epoch_idx)

        print('The IoU score of %d-view-input is %.4f\n' % (cfg.CONST.N_VIEWS_RENDERING, max_iou))

        return max_iou


def batch_test(cfg):
    import sys
    import os

    dir_name, _ = os.path.split(cfg.CONST.WEIGHTS)
    if True:
        log_file = os.path.join(dir_name, 'test_log_%s.txt' % dt.now().isoformat())
        f = open(log_file, 'w')
        sys.stdout = f

    view_num_list = [1, 2, 3, 4, 5, 8, 12, 16, 20]
    for view_num in view_num_list:
        cfg.CONST.N_VIEWS_RENDERING = view_num
        test_net(cfg)

    if True:
        f.close()
