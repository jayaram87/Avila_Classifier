import os, sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataAnalysisConfig
from src.entity.artifact_entity import DataIngestionArtifact
from pandas_profiling import ProfileReport

class DataAnalysis:
    def __init__(self, data_analysis_config: DataAnalysisConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_analysis_config = data_analysis_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def pandas_profiling(self):
        """
        Creates pandas profiling report for train and test datasets
        """
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_profile = ProfileReport(train_df, title="Train Data Profiling Report", explorative=True)
            test_profile = ProfileReport(test_df, title="Test Data Profiling Report", explorative=True)

            profile_page_file_path = self.data_analysis_config.profiling_page_file_path
            profile_page_dir = os.path.dirname(profile_page_file_path)
            os.makedirs(profile_page_dir, exist_ok=True)

            train_profile.to_file(profile_page_file_path)

            test_profile_name = os.path.basename(profile_page_file_path).replace('train', 'test')
            test_profile_page_file_path = os.path.join(profile_page_dir, test_profile_name)

            test_profile.to_file(test_profile_page_file_path) 
        except Exception as e:
            raise CustomException(e, sys) from e

    def complete_analysis(self) -> None:
        try:
            self.pandas_profiling()
            logging.info(f'Pandas Profiling train and test reports created')
        except Exception as e:
            raise CustomException(e, sys) from e