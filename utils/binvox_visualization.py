#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def get_volume_views(volume, save_dir, n_itr, angles):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    volume = volume.squeeze().__ge__(0.5)
    
    views = []
    for angle in angles:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.view_init(elev=angle[0], azim=angle[1])  # Set the elevation and azimuth angle
        ax.voxels(volume, edgecolor="k")

        save_path = os.path.join(save_dir, 'voxels-%06d-angle-%03d-%03d.png' % (n_itr, angle[0], angle[1]))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        views.append(cv2.imread(save_path))
    return views
