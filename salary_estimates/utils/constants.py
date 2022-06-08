import os

# Image path settings
DATA_DIRECTORY = 'data'
TRAIN_FEATURES = 'train_features.csv'
TEST_FEATURES = 'test_features.csv'
TRAIN_SALARIES = 'train_salaries.csv'
CLEANED_DATA = 'cleaned_data.csv'
TEST_DATA = 'test_data.csv'
PREDICTION_DATA = 'test_salaries.csv'
MODEL_FOLDER = "data/model-de0bb60f402f46b4a4bf25948c0a3d3c-1200-3-0.4"

TRAIN_FEATURES_PATH = os.path.join(DATA_DIRECTORY, TRAIN_FEATURES)
TRAIN_SALARIES_PATH = os.path.join(DATA_DIRECTORY, TRAIN_SALARIES)
TEST_FEATURES_PATH = os.path.join(DATA_DIRECTORY, TEST_FEATURES)

CLEANED_DATA_PATH = os.path.join(DATA_DIRECTORY, CLEANED_DATA)
TEST_DATA_PATH = os.path.join(DATA_DIRECTORY, TEST_DATA)
PREDICTIONS_PATH = os.path.join(DATA_DIRECTORY, PREDICTION_DATA)

