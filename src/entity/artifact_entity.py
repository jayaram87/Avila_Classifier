from collections import namedtuple

DataIngestionArtifact = namedtuple('DataIngestionArtifact', ['train_file_path', 'test_file_path', 'data_ingested', 'msg'])

DataValidationArtifact = namedtuple('DataValidationArtifact', ['schema_file_path', 'report_file_path', 'report_page_file_path', 'data_validated', 'msg'])

DataTransformationArtifact = namedtuple('DataTransformationArtifact', ['pca_train_file_path', 'pca_test_file_path', 'pca_sample_train_file_path', 'no_pca_sample_train_file_path', 'train_data_file_path', 'test_data_file_path', 'labelencoder', 'pca', 'transformed', 'msg'])

ModelArtifact = namedtuple('ModelArtifact', ['best_model', 'score', 'metric', 'accepted'])