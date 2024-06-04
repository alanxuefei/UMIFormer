#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import os
import torch
import numpy as np
from utils import logging
import utils.data_loaders
import utils.data_transforms
import utils.helpers
from tensorboardX import SummaryWriter
from datetime import datetime as dt
import tempfile


def modify_lr_strategy(cfg, current_epoch):
    if current_epoch <= cfg.TRAIN.WARMUP:
        raise ValueError('current_epoch <= cfg.TRAIN.WARM_UP; Please train from scratch!')
    
    if cfg.TRAIN.LR_scheduler == 'ExponentialLR':
        init_lr_list = \
            [cfg.TRAIN.ENCODER_LEARNING_RATE, cfg.TRAIN.DECODER_LEARNING_RATE, cfg.TRAIN.MERGER_LEARNING_RATE]
        current_epoch_lr_list = \
            [init_lr * (cfg.TRAIN.EXPONENTIALLR.SCHEDULE_FACTOR ** (current_epoch - cfg.TRAIN.WARM_UP))
             for init_lr in init_lr_list]
        cfg.TRAIN.ENCODER_LEARNING_RATE, cfg.TRAIN.DECODER_LEARNING_RATE, cfg.TRAIN.MERGER_LEARNING_RATE \
            = current_epoch_lr_list
    elif cfg.TRAIN.LR_scheduler == 'MilestonesLR':
        milestone_lists = [cfg.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES, cfg.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES]
        init_lr_list = [cfg.TRAIN.ENCODER_LEARNING_RATE, cfg.TRAIN.DECODER_LEARNING_RATE]
        if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
            milestone_lists.append(cfg.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES)
            init_lr_list.append(cfg.TRAIN.MERGER_LEARNING_RATE)
        current_milestone_list = []
        current_epoch_lr_list = []
        for milestones, init_lr in zip(milestone_lists, init_lr_list):
            milestones = np.array(milestones) - current_epoch
            init_lr = init_lr * cfg.TRAIN.MILESTONESLR.GAMMA ** len(np.where(milestones <= 0)[0])
            milestones = list(milestones[len(np.where(milestones <= 0)[0]):])
            current_milestone_list.append(milestones)
            current_epoch_lr_list.append(init_lr)
        cfg.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES = current_milestone_list[0]
        cfg.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES = current_milestone_list[1]
        cfg.TRAIN.ENCODER_LEARNING_RATE = current_epoch_lr_list[0]
        cfg.TRAIN.DECODER_LEARNING_RATE = current_epoch_lr_list[1]
        if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
            cfg.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES = current_milestone_list[2]
            cfg.TRAIN.MERGER_LEARNING_RATE = current_epoch_lr_list[2]
    else:
        raise ValueError(f'{cfg.TRAIN.LR_scheduler} is not supported.')

    return cfg


def load_data(cfg):
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
        utils.data_transforms.normalize
    ])
    # pix3d
    # train_transforms = utils.data_transforms.Compose([
    #     utils.data_transforms.RandomRotation(cfg.TRAIN.RANDOM_ROTATION),
    #     utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
    #     utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
    #     utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
    #     utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
    #     utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    #     utils.data_transforms.RandomFlip(),
    #     utils.data_transforms.RandomPermuteRGB(),  # not for Pix3D
    #     utils.data_transforms.ToTensor(),
    # ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
        utils.data_transforms.normalize
    ])

    # Set up data loader
    train_dataset, _ = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms)
    val_dataset, val_file_num = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.CONST.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True)

    return train_data_loader,  None, val_data_loader, val_file_num


def setup_network(cfg, encoder, decoder, merger):
    # Set up networks
    logging.info('Parameters in Encoder: %d.' % (utils.helpers.count_parameters(encoder)))
    logging.info('Parameters in Decoder: %d.' % (utils.helpers.count_parameters(decoder)))
    logging.info('Parameters in Merger: %d.' % (utils.helpers.count_parameters(merger)))

    # Initialize weights of networks
    decoder.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)

    # set sync bn
    if cfg.TRAIN.SYNC_BN:
        if True:
            print('Setting sync_batchnorm ...')
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
        if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
            merger = torch.nn.SyncBatchNorm.convert_sync_batchnorm(merger)
    else:
        if True:
            print('Without sync_batchnorm')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
        merger = merger.to(device)

    # Use DataParallel if more than one GPU is available
    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)
        if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
            merger = torch.nn.DataParallel(merger)
    
    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    
    if cfg.TRAIN.RESUME_TRAIN and 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % cfg.CONST.WEIGHTS)
        checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=device)
        init_epoch = checkpoint['epoch_idx'] + 1
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
            merger.load_state_dict(checkpoint['merger_state_dict'])
        
        logging.info('Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
                     (init_epoch, best_iou, best_epoch))
        
        # resume the learning-rate strategy
        cfg = modify_lr_strategy(cfg, init_epoch)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), 'initial_weights.pth')
        checkpoint = {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'merger_state_dict': merger.state_dict() if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS else None
        }
        
        if True:
            torch.save(checkpoint, checkpoint_path)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
            merger.load_state_dict(checkpoint['merger_state_dict'])
        
        if True:
            if os.path.exists(checkpoint_path) is True:
                os.remove(checkpoint_path)

    return init_epoch, best_iou, best_epoch, encoder, decoder, merger, cfg


def setup_writer(cfg):
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    print(output_dir)
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
    return train_writer, val_writer


def solver(cfg, encoder, decoder, merger):
    encoder_solver = torch.optim.AdamW(encoder.parameters(),
                                       lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        
    decoder_solver = torch.optim.AdamW(decoder.parameters(),
                                       lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    
    if cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
        return encoder_solver, decoder_solver
    else:
        merger_solver = torch.optim.AdamW(merger.parameters(),
                                          lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        return encoder_solver, decoder_solver, merger_solver
