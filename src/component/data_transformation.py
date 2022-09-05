import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SVMSMOTE
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.util import save_array_data, save_object
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact

    def pca_transform(self, data: np.array):
        """
        feature engineers the dataset using atleast 80% explained variance of the original data
        """
        try:
            for i in range(1, 6):
                pca = PCA(n_components=i)
                pca.fit(data)
                if sum(pca.explained_variance_ratio_) >= 0.8:
                    return pca.transform(data), pca
            return data, None
        except Exception as e:
            raise CustomException(e, sys) from e

    def encoding(self, data: np.array):
        """
        Label encodes the target variable
        """
        try:
            le = LabelEncoder()
            y = le.fit_transform(data)
            return y, le
        except Exception as e:
            raise CustomException(e, sys) from e

    def smotesvm(self, x, y, label_encoder):
        """
        Oversamples the miniority classes
        """
        try:
            labels = label_encoder.transform(['Y', 'C', 'W', 'B'])
            classes = {
                labels[0]: 380, 
                labels[1]: 350, 
                labels[2]: 350, 
                labels[3]: 500
            }
            sm = SVMSMOTE(sampling_strategy=classes, random_state=42, m_neighbors=3, k_neighbors=3)
            X_res, y_res = sm.fit_resample(x, y)
            return X_res, y_res
        except Exception as e:
            raise CustomException(e, sys) from e

    def data_transformed(self) -> DataTransformationArtifact:
        """
        Returns a transformation artifact with all the pca, non_pca transformed data with samapling
        'labelencoder', 'transformed', 'msg'])
        """
        try:
            logging.info(f'data transformation starts....')
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file = os.path.join(transformed_train_dir, 'train.npz')
            test_file = os.path.join(transformed_test_dir, 'test.npz')
            pca_train_file = os.path.join(transformed_train_dir, 'pca_train.npz')
            pca_test_file =  os.path.join(transformed_test_dir, 'pca_test.npz')
            pca_sample_train_file = os.path.join(transformed_train_dir, 'pca_sample_train.npz')
            no_pca_sample_train_file = os.path.join(transformed_train_dir, 'no_pca_sample_train.npz')
            le_file = os.path.join(transformed_train_dir, 'labelencoder.pkl')
            pca_file = os.path.join(transformed_train_dir, 'pca.pkl')


            train_x = train_df.iloc[:,:-1].values
            train_y = train_df.iloc[:,-1].values
            test_x = test_df.iloc[:,:-1].values
            test_y = test_df.iloc[:,-1].values

            pca_train_x, pca_obj = self.pca_transform(train_x)
            if pca_obj is not None:
                pca_test_x = pca_obj.transform(test_x)
            else:
                pca_test_x = test_x

            train_y, encoder = self.encoding(train_y)
            test_y = encoder.transform(test_y)

            pca_train = np.c_[pca_train_x, train_y]
            pca_test = np.c_[pca_test_x, test_y]

            save_array_data(train_file, np.c_[train_x, train_y])
            save_array_data(test_file, np.c_[test_x, test_y])
            save_array_data(pca_train_file, pca_train)
            save_array_data(pca_test_file, pca_test)
            save_object(le_file, encoder)
            if pca_obj is not None: save_object(pca_file, pca_obj)

            # smote svm training data oversampling for minority classes
            smote_train_pca = np.c_[self.smotesvm(pca_train_x, train_y, encoder)]
            smote_train = np.c_[self.smotesvm(train_x, train_y, encoder)]

            save_array_data(pca_sample_train_file, smote_train_pca)
            save_array_data(no_pca_sample_train_file, smote_train)

            data_transform_artifact = DataTransformationArtifact(
                pca_train_file_path = pca_train_file,
                pca_test_file_path = pca_test_file,
                pca_sample_train_file_path = pca_sample_train_file,
                no_pca_sample_train_file_path = no_pca_sample_train_file,
                train_data_file_path = train_file,
                test_data_file_path = test_file,
                labelencoder = le_file,
                pca = pca_file,
                transformed = True,
                msg = f'Data transformation completed'
            )

            logging.info(f'Data transformation completed....')
            return data_transform_artifact

        except Exception as e:
            raise CustomException(e, sys) from e