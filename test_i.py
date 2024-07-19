#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Import binvox_rw if you have it in the current directory
import utils.binvox_rw

# Import the get_volume_views function from binvox_visualization.py
from utils.binvox_visualization import get_volume_views

# Proper package import for the model and configuration
from models.voxel_net.voxnet import VoxNet
from config_i import cfg

def load_binvox_model(filepath):
    with open(filepath, 'rb') as f:
        model = utils.binvox_rw.read_as_3d_array(f)
    return model.data

def plot_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
    for ax, img_path in zip(axes, images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def plot_voxel_views(pred_voxel_images, gt_voxel_images):
    num_angles = len(pred_voxel_images)
    fig, axes = plt.subplots(2, num_angles, figsize=(15, 5))
    for i, (pred_img, gt_img) in enumerate(zip(pred_voxel_images, gt_voxel_images)):
        axes[0, i].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Predicted {i+1}')
        
        axes[1, i].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Ground Truth {i+1}')
    plt.show()

def visualize_voxel_with_ground_truth(model, image_path, ground_truth_path, save_dir, angles):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Change to (1, C, H, W)

    # Load the ground truth voxel model
    ground_truth = load_binvox_model(ground_truth_path)

    # Run the model to get the predicted voxel
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()
        pred_voxel = model(image)
        pred_voxel = torch.sigmoid(pred_voxel).squeeze().cpu().numpy()  # Apply threshold
        pred_voxel = (pred_voxel > 0.9) & (pred_voxel <= 1)

    ground_truth = ground_truth > 0.9

    # Visualize the predicted voxel
    pred_voxel_images = get_volume_views(pred_voxel, save_dir, 0, angles)

    # Visualize the ground truth voxel
    gt_voxel_images = get_volume_views(ground_truth, save_dir, 1, angles)

    # Plot both in one figure
    plot_voxel_views(pred_voxel_images, gt_voxel_images)

def main():
    root_dir = "./1a0bc9ab92c915167ae33d942430658c"
    model_path = "/workspace/output_i/model_epoch_360.pth"

    # Load the model
    model = VoxNet(cfg)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Path to the image
    image_path = os.path.join(root_dir, 'rendering', '00.png')
    # Path to the ground truth voxel model
    ground_truth_path = os.path.join(root_dir, 'Vox32', 'model.binvox')

    if os.path.exists(image_path) and os.path.exists(ground_truth_path):
        # Save and display the voxel visualization
        save_dir = os.path.join(root_dir, 'voxel_visualizations')
        angles = [(45, 0), (45, 45), (45, 90)]  # Three angles at 45-degree increments
        visualize_voxel_with_ground_truth(model, image_path, ground_truth_path, save_dir, angles)
    else:
        print(f"Image or ground truth voxel model not found at specified paths")

    # Path to the images
    rendering_dir = os.path.join(root_dir, 'rendering')
    if os.path.exists(rendering_dir):
        images = [os.path.join(rendering_dir, f"{i:02}.png") for i in range(24)]
        images = [img for img in images if os.path.exists(img)]
        plot_images(images)
    else:
        print(f"Rendering directory not found at {rendering_dir}")

if __name__ == '__main__':
    main()
