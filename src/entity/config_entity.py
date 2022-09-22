from collections import namedtuple

DataIngestionConfig = namedtuple('DataIngestionConfig', ['data_download_url', 'raw_data_dir', 'zip_download_dir', 'ingested_train_data_dir',
'ingested_test_data_dir'])

DataValidationConfig = namedtuple('DataValidationConfig', ['schema_file_path', 'report_file_path', 'report_page_file_path'])

DataAnalysisConfig = namedtuple('DataAnalysisConfig', ['profiling_page_file_path'])

DataTransformationConfig = namedtuple('DataTransformationConfig', ['transformed_train_dir', 'transformed_test_dir'])

ModelConfig = namedtuple('ModelConfig', ['model_file_path', 'score_path'])

TrainPipelineConfig = namedtuple('TrainPipelineConfig', ['artifact_dir'])