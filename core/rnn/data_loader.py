import numpy as np
import os
from os.path import join
import sklearn
import logging
import json
import codecs
from sklearn.model_selection import train_test_split

from core.utils import data_utils
from core.utils import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class PairTextDataset(object):

    def __init__(self, file_dir, seed, shuffle, max_sequence_length, max_key_sequence_length, batch_size, multiple):
        self.file_dir = file_dir

        # load data
        logger.info('loading training pairs...')
        self.msl = max_sequence_length
        self.mksl = max_key_sequence_length
        self.train_data = json.load(codecs.open(join(settings.VENUE_DATA_DIR, 'train.txt'), 'r', 'utf-8'))
        self.vocab = {}
        self.vocab_size = self.split_and_tokenize()
        self.stop_list = []
        self.batch_size = batch_size
        with codecs.open(join(settings.VENUE_DATA_DIR, 'stoplist.txt'), 'r', 'utf-8') as f:
            for word in f.readlines():
                self.stop_list.append(word[:-1])
        self.labels = []
        self.mag = []
        self.aminer = []
        self.keyword_mag = []
        self.keyword_aminer = []
        self.length_mag = []
        self.length_aminer = []
        self.jaccard = []
        self.inverse_pairs = []
        for pair in self.train_data:
            len_mag, len_aminer, keyword_mag, keyword_aminer, jaccard, inverse_pairs = self.preprocess(pair[1], pair[2])
            pair[1] = pair[1] + [0] * (self.msl - len(pair[1])) if len(pair[1]) <= self.msl else pair[1][:self.msl]
            pair[2] = pair[2] + [0] * (self.msl - len(pair[2])) if len(pair[2]) <= self.msl else pair[2][:self.msl]
            self.labels.append(pair[0])
            self.mag.append(pair[1])
            self.aminer.append(pair[2])
            self.length_mag.append(len_mag)
            self.length_aminer.append(len_aminer)
            self.keyword_mag.append(keyword_mag)
            self.keyword_aminer.append(keyword_aminer)
            self.jaccard.append([np.float32(jaccard)] * (multiple * 2))
            self.inverse_pairs.append([np.float32(inverse_pairs)] * multiple)
        self.labels, self.mag, self.aminer, self.length_mag, self.length_aminer, self.keyword_mag, self.keyword_aminer, self.jaccard, self.inverse_pairs = np.array(
            self.labels), np.array(self.mag), np.array(self.aminer), np.array(self.length_mag), np.array(
            self.length_aminer), np.array(self.keyword_mag), np.array(self.keyword_aminer), np.array(
            self.jaccard), np.array(self.inverse_pairs)
        logger.info('training pairs loaded')

        if shuffle:
            self.mag, self.aminer, self.labels = sklearn.utils.shuffle(self.mag, self.aminer, self.labels,
                                                                       random_state=seed)
        self.n_pairs = len(self.labels)
        logger.info('all pairs count %d', self.n_pairs)

    def split_and_tokenize(self):
        for i, pair in enumerate(self.train_data.copy()):
            seq1 = pair[1].split(' ')
            seq2 = pair[2].split(' ')
            pass
            for j, w in enumerate(seq1.copy()):
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab) + 1
                seq1[j] = self.vocab[w]
            for j, w in enumerate(seq2.copy()):
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab) + 1
                seq2[j] = self.vocab[w]
            self.train_data[i] = [pair[0], seq1, seq2]
        return len(self.vocab)

    def preprocess(self, seq1, seq2, use_stop_word=False):
        overlap = set(seq1).intersection(seq2)
        jaccard = len(overlap) / (len(seq1) + len(seq2) - len(overlap))
        # stop_word = set(seq1).union(seq2).difference(overlap) if not use_stop_word else None
        # keyword_seq1 = self.remove_stop_word(seq1, stop_word)
        # keyword_seq2 = self.remove_stop_word(seq2, stop_word)
        inverse_pairs, keyword_seq1, keyword_seq2 = self.compute_inverse_pairs(seq1, seq2, overlap)
        return len(seq1), len(seq2), keyword_seq1, keyword_seq2, jaccard, inverse_pairs

    def remove_stop_word(self, seq, stop_word=None):
        s = []
        stop_list = self.stop_list if not stop_word else stop_word
        for word in seq:
            if word not in stop_list:
                s.append(word)
        return s + [0] * (self.mksl - len(s)) if len(s) <= self.mksl else s[:self.mksl]

    def compute_inverse_pairs(self, seq1, seq2, overlap):
        look_up = {}
        new_seq1 = []
        new_seq2 = []
        for w in seq1:
            if w in overlap:
                look_up[w] = len(look_up) + 1
                new_seq1.append(look_up[w])
        for w in seq2:
            if w in overlap:
                new_seq2.append(look_up[w])
        result = 0
        for i in range(len(new_seq2)):
            for j in range(i, len(new_seq2)):
                if new_seq2[j] < i + 1:
                    result -= 1
        return result, \
               new_seq1 + [0] * (self.mksl - len(new_seq1)) if len(new_seq1) <= self.mksl else new_seq1[:self.mksl], \
               new_seq2 + [0] * (self.mksl - len(new_seq2)) if len(new_seq2) <= self.mksl else new_seq2[:self.mksl]

    def split_dataset(self, test_size):
        train = {'mag': None, 'aminer': None, 'keyword_mag': None, 'keyword_aminer': None, 'jaccard': None,
                 'inverse': None, 'labels': None}
        test = train.copy()
        train['mag'], test['mag'], train['aminer'], test['aminer'], train['keyword_mag'], test['keyword_mag'], train[
            'keyword_aminer'], test['keyword_aminer'], train['jaccard'], test['jaccard'], train['inverse'], test[
            'inverse'], train['labels'], test['labels'] = train_test_split(self.mag, self.aminer, self.keyword_mag,
                                                                           self.keyword_aminer, self.jaccard,
                                                                           self.inverse_pairs, self.labels,
                                                                           test_size=test_size)
        return DataLoader(self.batch_size, train), DataLoader(self.batch_size, test)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        return self.mag[idx], self.aminer[idx], self.jaccard[idx], self.keyword_mag[idx], self.keyword_aminer[idx], \
               self.inverse_pairs[idx], self.labels[idx]


class DataLoader(object):
    def __init__(self, batch_size, data: dict):
        self.batch_size = batch_size
        self.data = data

    def __iter__(self):
        N = len(self.data['labels'])
        iters = N // self.batch_size + 1
        for i in range(iters):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, N)
            yield np.array(self.data['mag'][start:end]), np.array(self.data['aminer'][start:end]), np.array(
                self.data['jaccard'][start:end]), np.array(self.data['keyword_mag'][start:end]), np.array(
                self.data['keyword_aminer'][start:end]), np.array(self.data['inverse'][start:end]), np.array(
                self.data['labels'][start:end])
