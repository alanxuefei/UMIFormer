from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset Config
__C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/ShapeNet_very_small.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = './datasets/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = './datasets/ShapeNetVox32/%s/%s/model.binvox'

# Dataset
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'ShapeNet'  # 'ShapeNetChairRFC'
__C.DATASET.TEST_DATASET = 'ShapeNet'  # 'Pix3D'

# Common
__C.CONST = edict()
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 224  # Image width for input
__C.CONST.IMG_H = 224  # Image height for input
__C.CONST.CROP_IMG_W = 128  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H = 128  # Dummy property for Pascal 3D
__C.CONST.BATCH_SIZE_PER_GPU = 8  # 16  # for train only
__C.CONST.N_VIEWS_RENDERING = 23
__C.CONST.NUM_WORKER = 20  # number of data workers
__C.CONST.WEIGHTS = './pths/UMIFormer.pth'

# Directories
__C.DIR = edict()
__C.DIR.OUT_PATH = './output_i/'

# Model
__C.MODEL = edict()
__C.MODEL.BACKBONE = 'resnet50'  # Example backbone, change as necessary

# Add VOXEL_HEAD configuration
__C.MODEL.VOXEL_HEAD = edict()
__C.MODEL.VOXEL_HEAD.NUM_CONV = 4
__C.MODEL.VOXEL_HEAD.CONV_DIM = 256
__C.MODEL.VOXEL_HEAD.NORM = "" # Normalization method for the convolution layers. Options: "" (no norm), "GN"
__C.MODEL.VOXEL_HEAD.VOXEL_SIZE = 32
__C.MODEL.VOXEL_HEAD.LOSS_WEIGHT = 1.0
__C.MODEL.VOXEL_HEAD.CUBIFY_THRESH = 0.0
__C.MODEL.VOXEL_HEAD.VOXEL_ONLY_ITERS = 100

# Training
__C.TRAIN = edict()

# for MilestonesLR
__C.TRAIN.MILESTONESLR = edict()
__C.TRAIN.MILESTONESLR.LR_MILESTONES = [50, 120]
__C.TRAIN.MILESTONESLR.GAMMA = .1

__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.SYNC_BN = True
__C.TRAIN.NUM_EPOCHS = 150
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]

__C.TRAIN.LR_scheduler = 'MilestonesLR'  # 'ExponentialLR' or 'MilestonesLR'
__C.TRAIN.WARMUP = 0

# for ExponentialLR
__C.TRAIN.EXPONENTIALLR = edict()
__C.TRAIN.EXPONENTIALLR.SCHEDULE_FACTOR = 1

__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.SAVE_FREQ = 10  # weights will be overwritten every save_freq epoch
__C.TRAIN.SHOW_TRAIN_STATE = 500

__C.TRAIN.LOSS = 4  # 1 for 'bce'; 2 for 'dice'; 3 for 'ce_dice'; 4 for 'focal'

__C.TRAIN.TEST_AFTER_TRAIN = True

# Testing options
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.3, .4, .5, .6]
