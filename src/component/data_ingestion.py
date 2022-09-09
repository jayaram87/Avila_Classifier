import os, sys, shutil
from zipfile import ZipFile
import numpy as np
import urllib.request as req
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.config.configuration import Configuration
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            logging.info(f'<-----------Data Ingestion module started -------->')
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys) from e

    def download_data(self) -> str:
        """
        download the zip file from ucl repo and return the string formatted zip file path
        """
        try:
            download_url = self.data_ingestion_config.data_download_url
            zip_file_dir = self.data_ingestion_config.zip_download_dir

            if os.path.exists(zip_file_dir):
                shutil.rmtree(zip_file_dir)

            os.makedirs(zip_file_dir, exist_ok=True)

            file_name = os.path.basename(download_url)
            zip_file_path = os.path.join(zip_file_dir, file_name)
            
            logging.info(f'Downloading file from {download_url} to {zip_file_path}')
            req.urlretrieve(download_url, zip_file_path)
            
            return zip_file_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def extract_zip_file(self, file_path: str) -> None:
        """
        Extracts the zip file into raw data directory
        """
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                shutil.rmtree(raw_data_dir)

            os.makedirs(raw_data_dir, exist_ok=True)

            with ZipFile(file_path) as zip_file:
                zip_file.extractall(raw_data_dir)
            logging.info(f'Extracted zip file into {raw_data_dir}')
        
        except Exception as e:
            raise CustomException(e, sys) from e

    def train_test_dataset(self) -> DataIngestionArtifact:
        """Train
        Function creates the train and test dataframe and stores them as csv files and return the data ingestion artifact object
        """
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            raw_data_avila_dir = os.path.join(raw_data_dir, 'avila')
            train_file_name, test_file_name = [i for i in os.listdir(raw_data_avila_dir) if i.startswith('avila-t')]
            train_file_path = os.path.join(raw_data_avila_dir, train_file_name)
            test_file_path = os.path.join(raw_data_avila_dir, test_file_name)
            column_list = ['intercolumnar_distance', 'upper_margin', 'lower_margin', 'exploitation', 'row_number', 'modular_ratio', 'interlinear_spacing', 'weight', 'peak_number', 'ratio', 'class']

            train_df = pd.read_csv(train_file_path, names=column_list)
            test_df = pd.read_csv(test_file_path, names=column_list)

            train_file_name = os.path.basename(train_file_path.replace('.txt', '.csv'))
            test_file_name = os.path.basename(test_file_path.replace('.txt', '.csv'))

            train_ingested_file_path = os.path.join(self.data_ingestion_config.ingested_train_data_dir, train_file_name)
            test_ingested_file_path = os.path.join(self.data_ingestion_config.ingested_test_data_dir, test_file_name)

            if len(train_df) > 0:
                os.makedirs(self.data_ingestion_config.ingested_train_data_dir, exist_ok=True)
                logging.info(f'train dataset ingested into train ingestion directory {self.data_ingestion_config.ingested_train_data_dir}')
                train_df.to_csv(train_ingested_file_path, index=False)

            if len(test_df) > 0:
                os.makedirs(self.data_ingestion_config.ingested_test_data_dir, exist_ok=True)
                logging.info(f'test dataset ingested into train ingestion directory {self.data_ingestion_config.ingested_test_data_dir}')
                test_df.to_csv(test_ingested_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = train_ingested_file_path,
                test_file_path = test_ingested_file_path, 
                data_ingested = True,
                msg = 'Data ingestion completed')
            
            logging.info(f'Data ingestion artifact {data_ingestion_artifact}')
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


    def data_ingestion(self) -> DataIngestionArtifact:
        try:
            avila_zip_file = self.download_data()
            self.extract_zip_file(avila_zip_file)
            return self.train_test_dataset()
        except Exception as e:
            raise CustomException(e, sys) from e