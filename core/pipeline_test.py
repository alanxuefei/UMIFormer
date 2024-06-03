#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch
import os
import numpy as np
import json
from utils import logging

import utils.data_loaders
import utils.data_transforms
import utils.helpers


def load_data(cfg, test_data_loader=None, test_file_num=None):
    # Load taxonomies of dataset
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}
    
    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.ToTensor(),
            utils.data_transforms.normalize
        ])
        
        dataset, test_file_num = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg).get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True)
    return taxonomies, test_data_loader, test_file_num


def setup_network(cfg, encoder, decoder, merger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
        merger = merger.to(device)
    
    logging.info('Loading weights from %s ...' % cfg.CONST.WEIGHTS)
    checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=device)
    epoch_idx = checkpoint['epoch_idx']

    # Adjust state_dict keys if necessary
    def adjust_state_dict_keys(state_dict, prefix='module.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    encoder_state_dict = adjust_state_dict_keys(checkpoint['encoder_state_dict'])
    decoder_state_dict = adjust_state_dict_keys(checkpoint['decoder_state_dict'])
    merger_state_dict = adjust_state_dict_keys(checkpoint['merger_state_dict']) if 'merger_state_dict' in checkpoint else {}

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    if not cfg.NETWORK.MERGER.WITHOUT_PARAMETERS:
        merger.load_state_dict(merger_state_dict)

    return encoder, decoder, merger, epoch_idx

def combine_test_iou(test_iou, taxonomies_list, taxonomies, test_file_num):
    # Single GPU version does not require all_gather and rank checks
    if test_iou is None or taxonomies_list is None:
        logging.error("test_iou or taxonomies_list is None")
        return {}
    
    test_iou = test_iou.cpu().numpy()  # [sample_num, 4]
    taxonomies_list = taxonomies_list.cpu().numpy()  # [sample_num]
    combined_test_iou = {}
    for taxonomy_id, sample_iou in zip(taxonomies_list, test_iou):
        if taxonomies[taxonomy_id] not in combined_test_iou.keys():
            combined_test_iou[taxonomies[taxonomy_id]] = {'n_samples': 0, 'iou': []}
        combined_test_iou[taxonomies[taxonomy_id]]['n_samples'] += 1
        combined_test_iou[taxonomies[taxonomy_id]]['iou'].append(sample_iou)
    return combined_test_iou

def output(cfg, test_iou, taxonomies):
    mean_iou = []
    n_samples = 0
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
        n_samples += test_iou[taxonomy_id]['n_samples']
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('\n')
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        print('N/a', end='\t\t')
        
        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()

    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    return mean_iou
