from collections import namedtuple

DataIngestionArtifact = namedtuple('DataIngestionArtifact', ['train_file_path', 'test_file_path', 'data_ingested', 'msg'])

DataValidationArtifact = namedtuple('DataValidationArtifact', ['schema_file_path', 'report_file_path', 'report_page_file_path', 'data_validated', 'msg'])