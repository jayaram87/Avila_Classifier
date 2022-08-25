import os, sys
from datetime import datetime
from src.constant import *
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig, TrainPipelineConfig
from src.utils.util import read_yaml_file
from src.exception import CustomException
from src.logger import logging


class Configuration:
    def __init__(self, config_path: str = CONFIG_FILE_PATH, current_time_stamp: str = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}') -> None: 
        self.config = read_yaml_file(config_path)
        self.timestamp = current_time_stamp
        self.train_pipeline_config = self.get_train_pipeline_config()


    def get_train_pipeline_config(self) -> TrainPipelineConfig:
        """
        function that returns the training pipeline config object with artifact_dir
        """
        try:
            train_pipeline_config = self.config[TRAINING_PIPELINE_CONFIG]
            artifact_dir = os.path.join(ROOT_DIR, train_pipeline_config[TRAINING_PIPELINE_NAME], train_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR])
            pipeline_config = TrainPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f'Training pipelien config: {pipeline_config}')
            return pipeline_config
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        function returns data ingestion configuration details from config yaml file
        """
        try:
            artifact_dir = self.train_pipeline_config.artifact_dir
            data_ingestion_artifact_dir = os.path.join(artifact_dir, DATA_INGESTION_ARTIFACT_DIR, self.timestamp) # directory for each initial_timestamp
            data_ingestion_config = self.config[DATA_INGESTION_CONFIG]
            data_download_url = data_ingestion_config[DATA_INGESTION_URL]
            zip_file_path = os.path.join(data_ingestion_artifact_dir, data_ingestion_config[DATA_INGESTION_ZIP_DOWNLOAD_DIR])
            raw_data_dir = os.path.join(data_ingestion_artifact_dir, data_ingestion_config[DATA_INGESTION_RAWDATA_DIR])
            ingested_data_dir = os.path.join(data_ingestion_artifact_dir, data_ingestion_config[DATA_INGESTION_DIR])
            train_data_dir = os.path.join(ingested_data_dir, data_ingestion_config[DATA_INGESTION_TRAIN_DATA])
            test_data_dir = os.path.join(ingested_data_dir, data_ingestion_config[DATA_INGESTION_TEST_DATA])

            data_ingestion_config =  DataIngestionConfig(
                data_download_url = data_download_url, 
                raw_data_dir = raw_data_dir,
                zip_download_dir = zip_file_path,
                ingested_train_data_dir = train_data_dir,
                ingested_test_data_dir = test_data_dir )

            logging.info(f'data ingestion config {data_ingestion_config.data_download_url}')
            return data_ingestion_config
            
        except Exception as e:
            logging.error(f'{str(e)}')
            raise CustomException(e, sys) from e

