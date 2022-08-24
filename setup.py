from setuptools import setup, find_packages
from typing import List

project_name = 'src'
version = '0.0.2'
author = 'Jayaram Raja'
description = 'Project to classify different manuscrips using the available physical features'

def requirements_list() -> List[str]:
    '''
    functions reads the requirements.txt file with all the required packages and appends them into a list format for setup
    '''
    with open('requirements.txt', 'r') as file:
        packages = file.readlines()
    packages = [i.replace('\n', '') for i in packages if i != '-e .']
    return packages

setup(
    name = project_name,
    version = version,
    author = author,
    description = description,
    packages = find_packages(),
    install_requires = requirements_list()
)
