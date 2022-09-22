import logging
from datetime import datetime
import os, shutil
import pandas as pd

log_dir = 'avila_logging'

os.makedirs(log_dir, exist_ok=True)

current_time_stamp = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
log_file_name = f'log_{current_time_stamp}.log'

log_file_path = os.path.join(log_dir, log_file_name)

logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='[%(asctime)s]|:|%(levelname)s|:|%(lineno)d|:|%(filename)s|:|%(funcName)s()|:|%(message)s',
        level=logging.INFO  
    )

def log_df(file_path):
    """
    reads the log file and returns the dataframe with time stamp & messages string
    """
    data = []
    with open(file_path) as file:
        for line in file.readlines():
            data.append(line.split('|:|'))
    df = pd.DataFrame(data)
    df.columns = ['Time stamp', 'Log level', 'Line nbr', 'filename', 'funcname', 'message']
    df['log_msg'] = df['Time stamp'].astype(str) + ':$' + df['message']
    return df['log_msg']
