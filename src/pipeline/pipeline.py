import os, sys
import uuid
import pandas as pd
from datetime import datetime
from typing import List
#from threading import Thread
#from multiprocessing import Process
from collections import namedtuple
from src.config.configuration import Configuration
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelArtifact
from src.component.data_ingestion import DataIngestion
from src.component.data_validation import DataValidation
from src.component.data_analysis import DataAnalysis
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import Trainer
from src.logger import logging
from src.exception import CustomException

# used to store the experiment/project run details in a csv
EXPERIMENT_DIR_NAME = 'experiment'
EXPERIMENT_FILE_NAME = 'experiment.csv'

Experiment = namedtuple('Experiment' , ['id', 'initial_timestamp', 'artifact_timestamp', 'status', 'start_time', 'stop_time', 'execution_time', 'msg', 'file_path', 'model_file_path', 'le_file_path', 'accuracy', 'model_accepted'])

class Pipeline:
    # class variables
    experiment = Experiment(*([None] * 13)) # initializing all values to None
    experiment_file_path = None

    def __init__(self, config: Configuration) -> None:
        try:
            os.makedirs(config.train_pipeline_config.artifact_dir, exist_ok=True) # creating an artifact directory
            # creating a experiment csv file path ./artifact/experiment/experiment.csv
            Pipeline.experiment_file_path = os.path.join(config.train_pipeline_config.artifact_dir, EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            self.config = config
            #super().__init__(daemon=True, name='pipeline')
        except Exception as e:
            raise CustomException(e, sys) from e

    def save_experiment(self):
        """
        Saving the experiment object as a dictionary in the experiment csv file using dataframe object
        """
        try:
            if Pipeline.experiment.id != None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict = {key: [value] for key, value in experiment_dict.items()} # key would colname and value would be values
                experiment_dict.update({
                    'created_time_stamp': [datetime.now()],
                    'experiment_file': [os.path.basename(Pipeline.experiment.file_path)]}
                )
                df = pd.DataFrame(experiment_dict)
                os.makedirs(os.path.dirname(Pipeline.experiment.file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment.file_path):
                    df.to_csv(Pipeline.experiment.file_path, index=False, header=False, mode='a')
                else:
                    df.to_csv(Pipeline.experiment.file_path, index=False, header=True, mode='w')
            else:
                logging.info(f'first experiment')

        except Exception as e:
            raise CustomException(e, sys) from e

    def data_ingestion(self) -> DataIngestionArtifact:
        """
        Ingests the zip file from ucl repo and saves train/test csv datasets and returns data ingestion artifact
        """
        try:
            d_ingestion = DataIngestion(self.config.get_data_ingestion_config())
            return d_ingestion.data_ingestion() # returns a data ingestion artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def data_validation(self, data_ing_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Validates the ingested train and test data for null values, column names and dtypes and returns a DataValidation artifact
        """
        try:
            data_validation = DataValidation(self.config.get_data_validation_config(), data_ing_artifact)
            return data_validation.complete_validation()
        except Exception as e:
            raise CustomException(e, sys) from e

    def data_analysis(self, data_ing_artifact: DataIngestion) -> None:
        """
        Pandas profiling reports in html and json format for train and test dataset
        """
        try:
            analysis = DataAnalysis(self.config.get_data_analysis_config(), data_ing_artifact)
            analysis.complete_analysis()
        except Exception as e:
            raise CustomException(e, sys) from e

    def data_transformation(self, data_ing_artifact: DataIngestion, data_val_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            """
            Train data is transformed using pca, pac with smote sampling, non_pac smote sampling
            Test data is transformed using pca
            Class labels are transfomred using label encoder for xgboost library
            All the files are saved as numpy arrays
            """
            data_transform = DataTransformation(self.config.get_data_transformation_config(), data_ing_artifact, data_val_artifact)
            return data_transform.data_transformed()
        except Exception as e:
            raise CustomException(e, sys) from e

    def best_model(self, data_transform_artifact: DataTransformationArtifact) -> ModelArtifact:
        """
        Returns a best model artifact with model, test_score and evaluation metric
        Model config: return ModelConfig object with best model path
        """
        try:
            classifier = Trainer(self.config.get_model_config(), data_transform_artifact)
            return classifier.best_classifier()

        except Exception as e:
            raise CustomException(e, sys) from e

    def save_experiment(self):
        """
        Saving the metadata of the pipeline experiment
        """
        try:
            if Pipeline.experiment.id is not None:
                experiment = Pipeline.experiment
                experiment_hash = {key: [value] for key, value in experiment._asdict().items()}
                experiment_hash.update(
                    {'created_time_stamp' : [datetime.now()],
                    'experiment_file_name' : [os.path.basename(experiment.file_path)]}
                )
                report = pd.DataFrame(experiment_hash)
                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment_file_path):
                    report.to_csv(Pipeline.experiment_file_path, mode='a', header=False, index=False) 
                else:
                    report.to_csv(Pipeline.experiment_file_path, mode='w', header=True, index=False) 

        except Exception as e:
            raise CustomException(e, sys) from e

    def pipeline_run(self):
        try:
            print(Pipeline.experiment.status)
            if Pipeline.experiment.status:
                # experitment is already running
                logging.info(f'Training pipeline is already running')
                return Pipeline.experiment

            logging.info(f'Training pipeline is starting')
            id = str(uuid.uuid4())
            Pipeline.experiment = Experiment(
                id = id,
                initial_timestamp = self.config.timestamp,
                artifact_timestamp = self.config.timestamp, 
                status = True,
                start_time = datetime.now(),
                stop_time = None,
                execution_time = None,
                msg = 'Pipeline has started',
                file_path = Pipeline.experiment_file_path, 
                model_file_path = None,
                le_file_path = None,
                accuracy = None,
                model_accepted = None
            )
            # saving experiment
            self.save_experiment()

            # data ingestion module
            data_ingestion_artifact = self.data_ingestion()
            data_validation_artifact = self.data_validation(data_ingestion_artifact)
            self.data_analysis(data_ingestion_artifact)
            data_transformation_artifact = self.data_transformation(data_ingestion_artifact, data_validation_artifact)
            model_artifact = self.best_model(data_transformation_artifact) 

            logging.info(f'Model training, evaluation completed....')
            stop_time = datetime.now()

            Pipeline.experiment = Experiment(
                id = Pipeline.experiment.id,
                initial_timestamp = self.config.timestamp,
                artifact_timestamp = self.config.timestamp, 
                status = False,
                start_time = Pipeline.experiment.start_time,
                stop_time = stop_time,
                execution_time = stop_time - Pipeline.experiment.start_time,
                msg = 'Training pipeline is completed',
                file_path = Pipeline.experiment_file_path, 
                model_file_path = model_artifact.best_model,
                le_file_path = data_transformation_artifact.labelencoder,
                accuracy = model_artifact.score,
                model_accepted = model_artifact.accepted
            )
            print(Pipeline.experiment)
            logging.info(f'Pipeline experiment {Pipeline.experiment}')
            self.save_experiment()

        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self):
        try:
            self.pipeline_run()
        except Exception as e:
            raise CustomException(e, sys) from e