import os
from os.path import join, abspath, dirname


PROJ_DIR = join(abspath(dirname(__file__)), '..', '..')
DATA_DIR = join(PROJ_DIR, 'data')
AUTHOR_DATA_DIR = join(DATA_DIR, 'authors')
PAPER_DATA_DIR = join(DATA_DIR, 'papers')
VENUE_DATA_DIR = join(DATA_DIR, 'venues')

OUT_DIR = join(PROJ_DIR, 'out')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PAPER_DIR = join(OUT_DIR, 'papers')
os.makedirs(OUT_PAPER_DIR, exist_ok=True)

AUTHOR_TYPE = 0
PAPER_TYPE = 1
VENUE_TYPE = 2
