import os

# Paths
DATASET_PATH = 'dataset'
RESULTS_PATH = 'results'
MODELS_PATH = os.path.join(RESULTS_PATH, 'models')
PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')
METRICS_PATH = os.path.join(RESULTS_PATH, 'metrics')
REPORTS_PATH = os.path.join(RESULTS_PATH, 'reports')

# Dataset
NUM_CLASSES = 9
CLASS_NAMES = ['Vegetation', 'Textile Trash', 'Plastic', 'Paper', 
               'Miscellaneous Trash', 'Metal', 'Glass', 'Food Organics', 'Cardboard']

# Image settings
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Models to train
MODELS_TO_TRAIN = ['CustomCNN', 'ResNet50', 'VGG16', 'MobileNetV2', 'EfficientNetB0']

# Ensemble
ENSEMBLE_METHODS = ['voting', 'weighted', 'stacking']


# ============================================================
# Total Files: Just 7 Python files + requirements.txt + README
# ============================================================

# 1. config.py              - Configuration (above)
# 2. data_preparation.py    - Load and prepare dataset (~150 lines)
# 3. train_models.py        - Train all models (~200 lines)
# 4. ensemble.py            - Ensemble methods (~150 lines)
# 5. evaluate.py            - Evaluation and comparison (~200 lines)
# 6. predict.py             - Prediction script (~80 lines)
# 7. utils.py               - Helper functions (~100 lines)

# Total: ~900 lines across 7 files instead of 30+ files!