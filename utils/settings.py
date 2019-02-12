import os
from os.path import join, abspath, dirname


PROJ_DIR = join(abspath(dirname(__file__)), '..')
DATA_DIR = join(PROJ_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = join(PROJ_DIR, 'model')
OUT_DIR = join(PROJ_DIR, 'out')
os.makedirs(OUT_DIR, exist_ok=True)
OAG_SHARE_DIR = '/home/share/oag/'
LARGE_GRAPH_DIR = join(OAG_SHARE_DIR, 'large-cross-graph')
TRAIN_DIR = join(DATA_DIR, 'train')

SAVED_MODEL = join(DATA_DIR, 'saved_model')
TRAIN_DATA = join(DATA_DIR, 'train_data')
RESULT = join(DATA_DIR, 'result')
RAW_TRAIN_DATA = join(DATA_DIR, 'raw_train_data')
PAPER_DIR = join(DATA_DIR, 'papers')


AUTHOR_TYPE = 0
PAPER_TYPE = 1
VENUE_TYPE = 2
