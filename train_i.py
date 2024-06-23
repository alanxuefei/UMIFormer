#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.backends.cudnn
import torch.utils.data
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config_i import cfg

import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.logging

from pprint import pprint

from time import time

import core.pipeline_train as pipeline

# Proper package import for VoxNet
from models.voxel_net.voxnet import VoxNet

from utils.average_meter import AverageMeter
from utils.scheduler_with_warmup import GradualWarmupScheduler
from losses.losses import DiceLoss, CEDiceLoss, FocalLoss

def train_net(cfg):
    print("Starting training...")
    # Enable the inbuilt cuDNN auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=cfg.DIR.OUT_PATH)

    print("Loading data...")
    print('Use config:')
    pprint(cfg)
    # Load data
    train_data_loader, train_sampler, val_data_loader, val_file_num = pipeline.load_data(cfg)
    print("Data loaded.")

    # Load model
    print("Loading model...")
    model = VoxNet(cfg)
    if torch.cuda.is_available():
        model = model.cuda()
    print("Model loaded.")

    # Set up solver with a fixed learning rate
    initial_learning_rate =  1e-4
    solver = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)

    # Set up learning rate scheduler to decay learning rates dynamically
    if cfg.TRAIN.LR_scheduler == 'ExponentialLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(solver, cfg.TRAIN.EXPONENTIALLR.SCHEDULE_FACTOR)
    elif cfg.TRAIN.LR_scheduler == 'MilestonesLR':
        warm_up = 0 if cfg.TRAIN.RESUME_TRAIN else cfg.TRAIN.WARMUP
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            solver, milestones=[lr - warm_up for lr in cfg.TRAIN.MILESTONESLR.LR_MILESTONES],
            gamma=cfg.TRAIN.MILESTONESLR.GAMMA)
    else:
        raise ValueError(f'{cfg.TRAIN.LR_scheduler} is not supported.')

    if cfg.TRAIN.WARMUP != 0 and not cfg.TRAIN.RESUME_TRAIN:
        lr_scheduler = GradualWarmupScheduler(solver, multiplier=1, total_epoch=cfg.TRAIN.WARMUP,
                                              after_scheduler=lr_scheduler)

    # Set up loss functions
    if cfg.TRAIN.LOSS == 1:
        loss_function = torch.nn.BCELoss()
    elif cfg.TRAIN.LOSS == 2:
        loss_function = DiceLoss()
    elif cfg.TRAIN.LOSS == 3:
        loss_function = CEDiceLoss()
    elif cfg.TRAIN.LOSS == 4:
        loss_function = FocalLoss()

    # Training loop
    n_views_rendering = cfg.CONST.N_VIEWS_RENDERING
    for epoch_idx in range(0, cfg.TRAIN.NUM_EPOCHS):
        # Tick / tock
        epoch_start_time = time()

        # Batch average metrics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # Switch model to training mode
        model.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        print(f"Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS} started. Number of batches: {n_batches}")

        for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(train_data_loader):
             # Log taxonomy names and sample names
            print(f'Batch {batch_idx}: Taxonomy names: {taxonomy_names}')
            print(f'Batch {batch_idx}: Sample names: {sample_names}')
            
            # Print the shape of rendering_images before slicing
            print(f'Batch {batch_idx}: Shape of rendering_images before slicing: {rendering_images.shape}')
            print(f'Batch {batch_idx}: Shape of ground_truth_volumes: {ground_truth_volumes.shape}')
            
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)

            if torch.cuda.is_available():
                rendering_images = rendering_images.to(torch.cuda.current_device())
                ground_truth_volumes = ground_truth_volumes.to(torch.cuda.current_device())

            # Process each view separately
            for view_idx in range(n_views_rendering):
                single_view_images = rendering_images[:, view_idx, :, :, :]

                # Forward pass
                voxel_scores = model(single_view_images).squeeze(dim=1)
                
                # Apply sigmoid to voxel scores
                generated_volumes = torch.sigmoid(voxel_scores)
                # Loss
                loss = loss_function(generated_volumes, ground_truth_volumes)

                # Gradient descent
                solver.zero_grad()
                loss.backward()
                solver.step()

                loss = utils.helpers.reduce_value(loss)

                # Append loss to average metrics
                losses.update(loss.item())

                # Log loss to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch_idx * n_batches + batch_idx)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if batch_idx == 0 or (batch_idx + 1) % cfg.TRAIN.SHOW_TRAIN_STATE == 0:
                print(f'[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}][Batch {batch_idx + 1}/{n_batches}] '
                      f'BatchTime = {batch_time.val:.3f} (s) DataTime = {data_time.val:.3f} (s) Loss = {loss.item():.4f}')
                print(f'LearningRate: {lr_scheduler.optimizer.param_groups[0]["lr"]} | {n_views_rendering}_views_rendering')

        # Adjust learning rate
        # lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print(f'[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}] EpochTime = {epoch_end_time - epoch_start_time:.3f} (s) '
              f'Loss = {losses.avg:.4f}')

        # Save the model every 10 epochs
        if (epoch_idx + 1) % 10 == 0:
            model_save_path = os.path.join(cfg.DIR.OUT_PATH, f'model_epoch_{epoch_idx + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')

    # Save the final model
    final_model_save_path = os.path.join(cfg.DIR.OUT_PATH, 'model_final.pth')
    torch.save(model.state_dict(), final_model_save_path)
    print(f'Final model saved to {final_model_save_path}')

    # Close the TensorBoard writer
    writer.close()

# Example usage
if __name__ == '__main__':
    train_net(cfg)
