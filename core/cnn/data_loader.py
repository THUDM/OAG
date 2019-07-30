from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from collections import defaultdict as dd
import numpy as np
import sklearn
from torch.utils.data import Dataset

from core.utils import feature_utils
from core.utils import data_utils
from core.utils import settings

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class CNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, build_index_window, seed, shuffle):

        self.file_dir = file_dir
        self.build_index_window = build_index_window

        self.matrix_title_size = matrix_size1
        self.matrix_author_size = matrix_size2

        # load training pairs
        pos_pairs = data_utils.load_json(file_dir, 'pos-pairs-train.json')
        pos_pairs = [(p['c'], p['n']) for p in pos_pairs]
        neg_pairs = data_utils.load_json(file_dir, 'neg-pairs-train.json')
        neg_pairs = [(p['c'], p['n']) for p in neg_pairs]
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        pairs = pos_pairs + neg_pairs

        n_matrix = len(pairs) * 2
        self.X_title = np.zeros((n_matrix * 2, self.matrix_title_size, self.matrix_title_size))
        self.X_author = np.zeros((n_matrix * 2, self.matrix_author_size, self.matrix_author_size))
        self.Y = np.zeros(n_matrix * 2, dtype=np.long)
        count = 0
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                logger.info('pairs to matrices %d', i)
            cpaper, npaper = pair
            cur_y = labels[i]
            matrix1 = self.titles_to_matrix(cpaper['title'], npaper['title'])
            self.X_title[count] = feature_utils.scale_matrix(matrix1)
            matrix2 = self.authors_to_matrix(cpaper['authors'], npaper['authors'])
            self.X_author[count] = feature_utils.scale_matrix(matrix2)
            self.Y[count] = cur_y
            count += 1

            # transpose
            self.X_title[count] = feature_utils.scale_matrix(matrix1.transpose())
            self.X_author[count] = feature_utils.scale_matrix(matrix2.transpose())
            self.Y[count] = cur_y
            count += 1

        if shuffle:
            self.X_title, self.X_author, self.Y = sklearn.utils.shuffle(
                self.X_title, self.X_author, self.Y,
                random_state=seed
            )

        self.N = len(self.Y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X_title[idx], self.X_author[idx], self.Y[idx]

    def get_noisy_papers_test(self):
        return data_utils.load_json_lines(self.file_dir, 'noisy-papers-test.dat')


    def titles_to_matrix(self, title1, title2):
        twords1 = feature_utils.get_words(title1)[: self.matrix_title_size]
        twords2 = feature_utils.get_words(title2)[: self.matrix_title_size]

        matrix = -np.ones((self.matrix_title_size, self.matrix_title_size))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

    def authors_to_matrix(self, authors1, authors2):
        matrix = -np.ones((self.matrix_author_size, self.matrix_author_size))
        author_num = int(self.matrix_author_size/2)
        try:
            for i in range(author_num):
                row = 2 * i
                a1 = authors1[i].lower().split()
                first_name1 = a1[0][0]
                last_name1 = a1[-1][0]
                col = row
                a2 = authors2[i].lower().split()
                first_name2 = a2[0][0]
                last_name2 = a2[-1][0]
                matrix[row][col] = feature_utils.name_equal(first_name1, first_name2)
                matrix[row][col+1] = feature_utils.name_equal(first_name1, last_name2)
                matrix[row+1][col] = feature_utils.name_equal(last_name1, first_name2)
                matrix[row+1][col+1] = feature_utils.name_equal(last_name1, last_name2)
        except Exception as e:
            pass
        return matrix

    def get_id2cpapers(self):
        cpapers_train = data_utils.load_json_lines(self.file_dir, 'clean-papers-train.dat')
        cpapers_test = data_utils.load_json_lines(self.file_dir, 'clean-papers-test.dat')
        cpapers = cpapers_train + cpapers_test
        id2paper = {}
        for paper in cpapers:
            paper['id'] = str(paper['id'])
            pid = paper['id']
            id2paper[pid] = paper
        # data_utils.dump_json(id2paper, self.file_dir, 'clean-id2paper.json')
        return id2paper

    def build_cpapers_inverted_index(self):
        logger.info('build inverted index for cpapers')
        cpapers_train = data_utils.load_json_lines(self.file_dir, 'clean-papers-train.dat')
        cpapers_test = data_utils.load_json_lines(self.file_dir, 'clean-papers-test.dat')
        papers = cpapers_train + cpapers_test
        word2ids = dd(list)
        for paper in papers:
            pid = str(paper['id'])
            title = paper['title']
            words = feature_utils.get_words(title.lower(), window=self.build_index_window)
            for word in words:
                word2ids[word].append(pid)
        for word in word2ids:
            word2ids[word] = list(set(word2ids[word]))
        # data_utils.dump_json(word2ids, self.file_dir, 'clean-papers-inverted-index.json')
        logger.info('building inverted index completed')
        return word2ids

    def get_candidates_by_inverted_index(self, npaper, word2ids):
        title = npaper['title'].lower()
        words = feature_utils.get_words(title, window=self.build_index_window)
        cids_to_freq = dd(int)
        for word in words:
            if word in word2ids:
                cur_cids = word2ids[word]
                for cid in cur_cids:
                    cids_to_freq[cid] += 1
        sorted_items = sorted(cids_to_freq.items(), key=lambda kv: kv[1], reverse=True)[:20]
        cand_cids = [item[0] for item in sorted_items]
        return cand_cids


if __name__ == '__main__':
    dataset = CNNMatchDataset(file_dir=settings.PAPER_DATA_DIR,
                              matrix_size1=7, matrix_size2=4, build_index_window=5,
                              seed=42, shuffle=True)
