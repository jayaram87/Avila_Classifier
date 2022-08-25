import os
from datetime import datetime

# root directories constants
ROOT_DIR = os.getcwd()
CONFIG_DIR = 'config'
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

#PIPELIN directory constants
TRAINING_PIPELINE_CONFIG = 'training_pipeline_config'
TRAINING_PIPELINE_NAME = 'pipeline_name'
TRAINING_PIPELINE_ARTIFACT_DIR = 'artifact_dir'

#data ingestion constants
DATA_INGESTION_CONFIG = 'data_ingestion_config'
DATA_INGESTION_ARTIFACT_DIR = 'data_ingestion'
DATA_INGESTION_URL = 'data_download_url'
DATA_INGESTION_RAWDATA_DIR = 'raw_data_dir'
DATA_INGESTION_ZIP_DOWNLOAD_DIR = 'zip_download_dir'
DATA_INGESTION_DIR = 'ingested_data_dir'
DATA_INGESTION_TRAIN_DATA = 'ingested_train_data_dir'
DATA_INGESTION_TEST_DATA = 'ingested_test_data_dir'
