import os, sys
from this import d
import pandas as pd
import json
from functools import reduce
from src.utils.util import read_yaml_file
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from evidently import dashboard
from evidently.model_profile import Profile # for json report
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def data_files_exist(self) -> bool:
        try:
            logging.info(f'checking whether train and test data files exist')

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            both_files_exists = os.path.exists(train_file_path) and os.path.exists(test_file_path)
            logging.info(f'Does train and test files exist: {both_files_exists}')

            if not both_files_exists:
                raise Exception(f'Train file {train_file_path} or test file {test_file_path} does not exist')
            
            return both_files_exists
         
        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_dataset_schema(self) -> bool:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            validation_check = False
            column_check = 0
            null_value_check = 0
            dtype_value_check = 0
            column_names = read_yaml_file(self.data_validation_config.schema_file_path)['float_columns'] + [read_yaml_file(self.data_validation_config.schema_file_path)['target_column']]
            
            for file in [train_df, test_df]:
                column_check += sum(file.columns == column_names)
            
            for file in [train_df, test_df]:
                for col in file.columns:
                    if col == read_yaml_file(self.data_validation_config.schema_file_path)['target_column']:
                        dtype_value_check += file[col].dtype == 'object'
                    else:
                        dtype_value_check += file[col].dtype == 'float64'

            for file in [train_df, test_df]:
                null_value_check += ~file.isnull().any().any()

            vals = column_check + dtype_value_check + null_value_check
            print(vals)

            if vals == 46: # 22 col checks, 22 dtype checks, 2 null_value checks
                validation_check = True
                return validation_check
            else:
                raise Exception(f'check vals {vals} is not satisfactory, please check the data sets')

        except Exception as e:
            raise CustomException(e, sys) from e

    def drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            train_df.rename(columns={"class": "target"})
            test_df.rename(columns={"class": "target"})
            profile.calculate(train_df, test_df, column_mapping=None)

            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            with open(self.data_validation_config.report_file_path, 'w') as j_file:
                json.dump(report, j_file, indent=4)

            return report

        except Exception as e:
            raise CustomException(e, sys) from e

    def save_drift_report_page(self):
        """
        Save the generated drift report into a html page
        """
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            train_df.rename(columns={"class": "target"})
            test_df.rename(columns={"class": "target"})

            dashboard.calculate(train_df, test_df, column_mapping=None)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)

            dashboard.save(report_page_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e

    def is_data_drift_found(self):
        """
        Evidently dashboards report above the datasets
        """
        try:
            report = self.drift_report()
            self.save_drift_report_page()
            return True
        except Exception as e:
            raise CustomException(e, sys) from e

    def complete_validation(self) -> DataValidationArtifact:
        try:
            self.data_files_exist()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path = self.data_validation_config.schema_file_path,
                report_file_path = self.data_validation_config.report_file_path,
                report_page_file_path = self.data_validation_config.report_page_file_path,
                data_validated = True,
                msg = f'Data validation completed'
            )
            logging.info(f'Data validation artifact {data_validation_artifact}')
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e