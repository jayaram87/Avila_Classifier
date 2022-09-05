import yaml, os, sys
import numpy as np
import pickle
import json
from src.exception import CustomException
from src.constant import *

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a yaml file and returns a dictionary object
    """
    try:
        with open(file_path, 'rb') as file:
            dict_file = yaml.safe_load(file)
        return dict_file
    except Exception as e:
        raise CustomException(e, sys) from e

def save_array_data(file_path: str, arr: np.array):
    """
    Saves the data in numpy array format
    """ 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as nfile:
            np.save(nfile, arr)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_object(file_path: str, object):
    """
    saves the object in pickle object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as pfile:
            pickle.dump(object, pfile)
    except Exception as e:
        raise CustomException(e, sys) from e
