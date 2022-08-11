from yacs.config import CfgNode as CN

CFG = CN()

CFG.DATASET = CN()
CFG.DATASET.NAME = ''  # dataset name, refer to datas/__init__.py
CFG.DATASET.ROOT = ''  # dataset directory
CFG.DATASET.NUM_CLASSES = 0  # number of classes for classification
CFG.DATASET.TRANSFORM = CN()
CFG.DATASET.TRANSFORM.ADAPTIVE_CROP = CN()
CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.ORIGIN_HEIGHTS = []
CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.ORIGIN_WIDTHS = []
CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.TOPS = []
CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.LEFTS = []
CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.HEIGHTS = []
CFG.DATASET.TRANSFORM.ADAPTIVE_CROP.WIDTHS = []
CFG.DATASET.TRANSFORM.RESIZE = CN()
CFG.DATASET.TRANSFORM.RESIZE.HEIGHT = 0
CFG.DATASET.TRANSFORM.RESIZE.WIDTH = 0
CFG.DATASET.TRANSFORM.RANDOM_HORIZONTAL_FLIP = CN()
CFG.DATASET.TRANSFORM.RANDOM_HORIZONTAL_FLIP.PROBABILITY = 0.
CFG.DATASET.TRANSFORM.RANDOM_ROTATION = CN()
CFG.DATASET.TRANSFORM.RANDOM_ROTATION.DEGREE = 0
CFG.DATASET.TRANSFORM.BRIGHTNESS = 1.
CFG.DATASET.TRANSFORM.CONTRAST = 1.
CFG.DATASET.TRANSFORM.NORMALIZE = CN()
CFG.DATASET.TRANSFORM.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
CFG.DATASET.TRANSFORM.NORMALIZE.STD = [0.229, 0.224, 0.225]

CFG.DATALOADER = CN()
CFG.DATALOADER.BATCH_SIZE = 0  # batch size
CFG.DATALOADER.NUM_WORKERS = 0  # number of workers for PyTorch DataLoader

CFG.MODEL = CN()
CFG.MODEL.NAME = ''  # model name, refer to models/__init__.py

CFG.CRITERION = CN()
CFG.CRITERION.NAME = ''  # criterion name, refer to criterions/__init__.py
CFG.CRITERION.SMOOTHING = 0.

CFG.OPTIMIZER = CN()
CFG.OPTIMIZER.NAME = ''  # optimizer name, refer to optimizers/__init__.py
CFG.OPTIMIZER.LR = 0.  # basic learning rate
CFG.OPTIMIZER.MOMENTUM = 0.
CFG.OPTIMIZER.WEIGHT_DECAY = 0.

CFG.SCHEDULER = CN()
CFG.SCHEDULER.NAME = ''  # scheduler name, refer to scheduler/__init__.py
CFG.SCHEDULER.GAMMA = 0.
# one of CFG.SCHEDULER.BY_EPOCH and CFG.SCHEDULER.BY_ITERATION should be true
CFG.SCHEDULER.BY_EPOCH = False
CFG.SCHEDULER.BY_ITERATION = False

# both CFG.EPOCHS and CFG.ITERATIONS should be set
CFG.EPOCHS = 0
CFG.ITERATIONS = 0

CFG.freeze()
