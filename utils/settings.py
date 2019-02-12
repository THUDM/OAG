# coding = 'utf-8'

from os.path import join, abspath, dirname

PROJ_DIR = abspath(dirname(__file__))
DATA_DIR = join(PROJ_DIR, 'data')
MODEL_DIR = join(PROJ_DIR, 'model')
PREPROCESS_DIR = join(PROJ_DIR, 'preprocess')

SAVED_MODEL = join(DATA_DIR, 'saved_model')
TRAIN_DATA = join(DATA_DIR, 'train_data')
RESULT = join(DATA_DIR, 'result')
RAW_TRAIN_DATA = join(DATA_DIR, 'raw_train_data')
