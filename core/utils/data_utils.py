import io
from os.path import join
import json
import pickle

from torch.utils.data.sampler import Sampler

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_vectors(in_dir, fname):
    fin = io.open(join(in_dir, fname), 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for i, line in enumerate(fin):
        if i % 10000 == 0:
            logger.info('load %s, line %d', fname, i)
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def load_json_lines(fpath, fname):
    items = []
    with open(join(fpath, fname), 'r', encoding='utf-8') as rf:
        for line in rf:
            item = json.loads(line)
            items.append(item)
    return items


def dump_large_obj(obj, wfdir, wfname):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    with open(join(wfdir, wfname), 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def load_large_obj(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'rb') as rf:
        obj = pickle.load(rf)
        logger.info('%s loaded', rfname)
        return obj


class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired data points
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples