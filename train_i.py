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
    fixed_learning_rate = 0.001
    solver = torch.optim.Adam(model.parameters(), lr=fixed_learning_rate)

    # Set up BCELoss
    loss_function = torch.nn.BCELoss()

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
            # Print the shape of rendering_images before slicing
            print(f'Batch {batch_idx}: Shape of rendering_images before slicing: {rendering_images.shape}')
            
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
                voxel_scores = torch.sigmoid(voxel_scores)
                # Loss
                loss = loss_function(voxel_scores, ground_truth_volumes)

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
                print(f'LearningRate: {fixed_learning_rate} | {n_views_rendering}_views_rendering')

        # Tick / tock
        epoch_end_time = time()
        print(f'[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}] EpochTime = {epoch_end_time - epoch_start_time:.3f} (s) '
              f'Loss = {losses.avg:.4f}')

    # Close the TensorBoard writer
    writer.close()

# Example usage
if __name__ == '__main__':
    train_net(cfg)
