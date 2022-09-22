import json
import os, sys
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent
from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelArtifact
from src.entity.config_entity import ModelConfig
from src.utils.util import save_object
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score
import xgboost as xgb

class Trainer:
    def __init__(self, model_config: ModelConfig, data_transform_artifact: DataTransformationArtifact):
        try:
            self.model_config = model_config
            self.data_transform_artifact = data_transform_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def best_classifier(self) -> ModelArtifact:
        """
        Return the model artifact with the best model based on fbeta score on test data, test_score and roc-auc curve
        """
        try:
            #logging.info(f'Model training starting......')
            #pca_data_xtrain, pca_data_ytrain = self.load_data(self.data_transform_artifact.pca_sample_train_file_path)
            #pca_data_xtest, pca_data_ytest = self.load_data(self.data_transform_artifact.pca_test_file_path)
            #pca_sample_xtrain, pca_sample_ytrain = self.load_data(self.data_transform_artifact.pca_sample_train_file_path)
            x_train, y_train = self.load_data(self.data_transform_artifact.train_data_file_path)
            x_test, y_test = self.load_data(self.data_transform_artifact.test_data_file_path)
            sample_xtrain, sample_ytrain = self.load_data(self.data_transform_artifact.no_pca_sample_train_file_path)

            # multiprocessing will not be work well with logging write setup
            """
            with concurrent.futures.ProcessPoolExecutor() as executor:
                pca_xbg = executor.submit(self.model_training, args=(xgb.XGBClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'xgboost', True, False))
                pca_rf = executor.submit(self.model_training, args=(RandomForestClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'RandomForest', True, False))
                pca_gb = executor.submit(self.model_training, args=(GradientBoostingClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'GrandientBoost', True, False))
                pca_et = executor.submit(self.model_training, args=(ExtraTreesClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'ExtraTress', True, False))
                pca_sample_xbg = executor.submit(self.model_training, args=(xgb.XGBClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'xgboost', True, True))
                pca_sample_rf = executor.submit(self.model_training, args=(RandomForestClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'RandomForest', True, True))
                pca_sample_gb = executor.submit(self.model_training, args=(GradientBoostingClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'GrandientBoost', True, True))
                pca_sample_et = executor.submit(self.model_training, args=(ExtraTreesClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'ExtraTress', True, True))
                xbg = executor.submit(self.model_training, args=(xgb.XGBClassifier(), x_train, x_test, y_train, y_test, 'xgboost', False, False))
                rf = executor.submit(self.model_training, args=(RandomForestClassifier(), x_train, x_test, y_train, y_test, 'RandomForest', False, False))
                gb = executor.submit(self.model_training, args=(GradientBoostingClassifier(), x_train, x_test, y_train, y_test, 'GrandientBoost', False, False))
                et = executor.submit(self.model_training, args=(ExtraTreesClassifier(), x_train, x_test, y_train, y_test, 'ExtraTress', False, False))
                sample_xbg = executor.submit(self.model_training, args=(xgb.XGBClassifier(), sample_xtrain, x_test, y_train, sample_ytrain, 'xgboost', False, True))
                sample_rf = executor.submit(self.model_training, args=(RandomForestClassifier(), sample_xtrain, x_test, y_train, sample_ytrain, 'RandomForest', False, True))
                sample_gb = executor.submit(self.model_training, args=(GradientBoostingClassifier(), sample_xtrain, x_test, y_train, sample_ytrain, 'GrandientBoost', False, True))
                sample_et = executor.submit(self.model_training, args=(ExtraTreesClassifier(), sample_xtrain, x_test, y_train, sample_ytrain, 'ExtraTress', False, True))

                result = pd.DataFrame([pca_xbg.result(), pca_rf.result(), pca_gb.result(), pca_et.result(), pca_sample_xbg.result(), pca_sample_rf.result(), pca_sample_gb.result(), pca_sample_et.result(), xbg.result(), rf.result(), gb.result(), et.result(), sample_xbg.result(), sample_rf.result(), sample_gb.result(), sample_et.result()])
            """
            # With PCA fbeta scores were bad becoz the raw data was already standardized
            #pca_xbg = self.model_training(xgb.XGBClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'xgboost', True, False)
            #pca_rf = self.model_training(RandomForestClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'RandomForest', True, False)
            #pca_gb = self.model_training(GradientBoostingClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'GrandientBoost', True, False)
            #pca_et = self.model_training(ExtraTreesClassifier(), pca_data_xtrain, pca_data_xtest, pca_data_ytrain, pca_data_ytest, 'ExtraTress', True, False)
            #pca_sample_xbg = self.model_training(xgb.XGBClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'xgboost', True, True)
            #pca_sample_rf = self.model_training(RandomForestClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'RandomForest', True, True)
            #pca_sample_gb = self.model_training(GradientBoostingClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'GrandientBoost', True, True)
            #pca_sample_et = self.model_training(ExtraTreesClassifier(), pca_sample_xtrain, pca_data_xtest, pca_sample_ytrain, pca_data_ytest, 'ExtraTress', True, True)
            xbg = self.model_training(xgb.XGBClassifier(), x_train, x_test, y_train, y_test, 'xgboost', False, False)
            rf = self.model_training(RandomForestClassifier(), x_train, x_test, y_train, y_test, 'RandomForest', False, False)
            gb = self.model_training(GradientBoostingClassifier(), x_train, x_test, y_train, y_test, 'GrandientBoost', False, False)
            et = self.model_training(ExtraTreesClassifier(), x_train, x_test, y_train, y_test, 'ExtraTress', False, False)
            sample_xbg = self.model_training(xgb.XGBClassifier(), sample_xtrain, x_test, sample_ytrain, y_test, 'xgboost', False, True)
            sample_rf = self.model_training(RandomForestClassifier(), sample_xtrain, x_test, sample_ytrain, y_test, 'RandomForest', False, True)
            sample_gb = self.model_training(GradientBoostingClassifier(), sample_xtrain, x_test, sample_ytrain, y_test, 'GrandientBoost', False, True)
            sample_et = self.model_training(ExtraTreesClassifier(), sample_xtrain, x_test, sample_ytrain, y_test, 'ExtraTress', False, True)

            result = pd.DataFrame([xbg, rf, gb, et, sample_xbg, sample_rf, sample_gb, sample_et])

            best_model = result[result['test_fbeta_score'] == result['test_fbeta_score'].max()]['model'][0]
            best_params = result[result['test_fbeta_score'] == result['test_fbeta_score'].max()]['model_params'][0]
            sampling = result[result['test_fbeta_score'] == result['test_fbeta_score'].max()]['sampling'][0]

            
            if sampling:
                if best_model == 'xgboost':
                    final_model = xgb.XGBClassifier(**best_params)
                    final_model.fit(sample_xtrain, sample_ytrain)
                elif best_model == 'RandomForest':
                    final_model = RandomForestClassifier(**best_params)
                    final_model.fit(sample_xtrain, sample_ytrain)
                elif best_model == 'GrandientBoost':
                    final_model = GradientBoostingClassifier(**best_params)
                    final_model.fit(sample_xtrain, sample_ytrain)
                elif best_model == 'ExtraTress':
                    final_model = ExtraTreesClassifier(**best_params)
                    final_model.fit(sample_xtrain, sample_ytrain)
            else:
                if best_model == 'xgboost':
                    final_model = xgb.XGBClassifier(**best_params)
                    final_model.fit(x_train, y_train)
                elif best_model == 'RandomForest':
                    final_model = RandomForestClassifier(**best_params)
                    final_model.fit(x_train, y_train)
                elif best_model == 'GrandientBoost':
                    final_model = GradientBoostingClassifier(**best_params)
                    final_model.fit(x_train, y_train)
                elif best_model == 'ExtraTress':
                    final_model = ExtraTreesClassifier(**best_params)
                    final_model.fit(x_train, y_train)
            
            save_object(self.model_config.model_file_path, final_model)

            score, metric = self.evaluate(final_model, x_test, y_test)

            data = json.dumps({'test_score': score, 'metric': metric})

            with open(self.model_config.score_path, 'w+') as f:
                f.write(data)
            
            model_artifact = ModelArtifact(
                best_model = self.model_config.model_file_path, 
                score = self.model_config.score_path,
                accepted = f'Best model generated'
            )

            logging.info(f'Model training completed {model_artifact}')
            return model_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def model_training(self, classifier, x_train, x_test, y_train, y_test, model_spec, pca=False, sampling=False) -> Dict:
        """
        returns the model dictionary with the model, train and test fbeta score
        """
        try:
            scorer = make_scorer(fbeta_score, beta=1, average='weighted')
            model_dict = {}
            if model_spec in ['GrandientBoost', 'xgboost']:
                params = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'random_state': [1]
                }
                grid = GridSearchCV(classifier, param_grid=params, n_jobs=-1, scoring=scorer)
                grid.fit(x_train, y_train)
                model_dict['model'] = model_spec
                model_dict['model_params'] = grid.best_params_
                model_dict['pca'] = pca
                model_dict['sampling'] = sampling
                model_dict['Train_fbeta_score'] = grid.best_score_
                model_dict['test_fbeta_score'] = fbeta_score(y_test, grid.predict(x_test), beta=1, average='weighted')

            elif model_spec in ['RandomForest', 'ExtraTress']:
                params = {
                    'n_estimators': [50, 100],
                    'max_features': ['auto', 'log2'],
                    'random_state': [1]
                }
                grid = GridSearchCV(classifier, param_grid=params, n_jobs=-1, scoring=scorer)
                grid.fit(x_train, y_train)
                model_dict['model'] = model_spec
                model_dict['model_params'] = grid.best_params_
                model_dict['pca'] = pca
                model_dict['sampling'] = sampling
                model_dict['Train_fbeta_score'] = grid.best_score_
                model_dict['test_fbeta_score'] = fbeta_score(y_test, grid.predict(x_test), beta=1, average='weighted')
            
            return model_dict
        
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(self, model, x_test, y_test):
        """
        returns the fbeta score and the metric and stores the confusion matirx as jpg and json
        """
        try:
            metric = 'fbeta_score'
            score = round(fbeta_score(y_test, model.predict(x_test), beta=1, average='weighted'),2)
            logging.info(f'model was evaluated with fbeta score on the test data')
            return score, metric
        except Exception as e:
            raise CustomException(e, sys) from e


    def load_data(self, file_path: str):
        """
        Returns x and y data in numpy array
        """
        try:
            data = np.load(file_path)
            return data[:,:-1], data[:,-1]
        except Exception as e:
            raise CustomException(e, sys) from e