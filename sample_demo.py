import os
from src.pipeline.pipeline import Pipeline
from src.logger import logging
from src.config.configuration import Configuration

def demo():
    try:
        config_path = os.path.join('config', 'config.yaml')
        pipeline = Pipeline(Configuration(config_path))
        logging.info(f'Demo training pipeline started')
        pipeline.run() # starting
        logging.info(f'Demo training pipeline completed')
    except Exception as e:
        logging.error(f'{str(e)}')

if __name__ == '__main__':
    demo()

