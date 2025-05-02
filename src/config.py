import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 1024
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 100
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
LEARNING_RATE       = 0.0005
OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': LEARNING_RATE}
# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
SCHEDULER_PARAMS    = {'type': 'CosineAnnealingLR', 'T_max': NUM_EPOCHS, 'eta_min': 1e-6}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 16

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'MyNetwork'
# MODEL_NAME          = 'efficientnet_b0'
# MODEL_NAME          = 'efficientnet_b5'
# MODEL_NAME          = 'efficientnet_b2'
# MODEL_NAME          = 'efficientnet_b3'
# MODEL_NAME          = 'efficientnet_b4'
# MODEL_NAME          = 'efficientnet_b5'
# MODEL_NAME          = 'efficientnet_b6'
# MODEL_NAME          = 'efficientnet_b7'
# MODEL_NAME          = 'resnet18'
# MODEL_NAME          = 'resnet50'
# MODEL_NAME          = 'resnet101'
# MODEL_NAME          = 'swin_t'
# MODEL_NAME          = 'swin_s'
# MODEL_NAME          = 'swin_b'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [1]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
# WANDB_NAME          = f'{MODEL_NAME}-SOTA-lr{LEARNING_RATE}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME          = f'{MODEL_NAME}-baseline-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
