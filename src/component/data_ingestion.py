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
            logging(f'Extracted zip file into {raw_data_dir}')
        
        except Exception as e:
            raise CustomException(e, sys) from e


    def data_ingestion(self) -> DataIngestionArtifact:
        try:
            avila_zip_file = self.download_data()
            self.extract_zip_file(avila_zip_file)
        except Exception as e:
            raise CustomException(e, sys) from e