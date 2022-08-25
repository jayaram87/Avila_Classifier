import yaml, os, sys
import numpy
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

