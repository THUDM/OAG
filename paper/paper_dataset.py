from os.path import join
import os
import codecs
import json
from collections import defaultdict as dd
from utils import data_utils
from utils import settings


class PaperDataUtils:
    paper_dir = settings.PAPER_DIR
    pairs_dir = join(settings.DATA_DIR, 'pairs')
    inverted_index_dir = join(settings.DATA_DIR, 'inverted-index')

    def __init__(self, ii_window=5):
        self.ii_window = ii_window

    def construct_positive_paper_pairs(self, role, fold):
        fname = '{}-papers-{}-{}.dat'
        cpapers_fname = fname.format('clean', role, fold)
        cpapers = data_utils.load_json_lines(self.paper_dir, cpapers_fname)
        npapers_fname = fname.format('noisy', role, fold)
        npapers = data_utils.load_json_lines(self.paper_dir, npapers_fname)
        pos_pairs = list(zip(cpapers, npapers))
        pairs_fname = 'pos-pairs-{}-{}.json'.format(role, fold)
        self.dump_paper_pairs(pos_pairs, pairs_fname)
        print('pos_pairs', len(pos_pairs))
        return pos_pairs

    def dump_paper_pairs(self, paper_pairs, wfname):
        paper_pairs_refactor = []
        for pair in paper_pairs:
            cpaper, npaper = pair
            pair_refactor = {'c': cpaper, 'n': npaper}
            paper_pairs_refactor.append(pair_refactor)
        data_utils.dump_json(paper_pairs_refactor, self.pairs_dir, wfname)

    def load_train_neg_paper_pairs(self, fold):
        fname = 'train-neg-pairs-{}.json'.format(fold)
        neg_pairs = data_utils.load_json(self.pairs_dir, fname)
        neg_pairs = [(p['c'], p['n']) for p in neg_pairs]
        print('neg_pairs', len(neg_pairs))
        return neg_pairs

    def build_inverted_index(self, fold):
        print('build inverted index for cpapers: fold', fold)
        fname = 'clean-papers-test-{}.dat'.format(fold)
        papers = data_utils.load_json_lines(self.paper_dir, fname)
        word2ids = dd(list)
        for paper in papers:
            pid = str(paper['id'])
            title = paper['title']
            words = data_utils.get_words(title, window=self.ii_window)
            for word in words:
                word2ids[word].append(pid)
        for word in word2ids:
            word2ids[word] = list(set(word2ids[word]))
        data_utils.dump_json(word2ids, self.inverted_index_dir, 'clean-papers-test-ii-{}.json'.format(fold))
        print('complete building II')
        return word2ids

    def load_inverted_index(self, fold):
        rfpath = join(self.inverted_index_dir, 'clean-papers-test-ii-{}.json'.format(fold))
        if not os.path.isfile(rfpath):
            self.build_inverted_index(fold)
        with codecs.open(rfpath, 'r', encoding='utf-8') as rf:
            word2ids = json.load(rf)
            return word2ids

    def gen_id2papers(self, fold):
        fname = 'clean-papers-test-{}.dat'.format(fold)
        cpapers = data_utils.load_json_lines(settings.PAPER_DIR, fname)
        id2paper = {}
        for paper in cpapers:
            paper['id'] = str(paper['id'])
            pid = paper['id']
            id2paper[pid] = paper
        # dump_json(id2paper, self.paper_dir, 'clean-id2paper-test-{}.json'.format(fold))
        return id2paper

    def load_id2papers(self, fold):
        return data_utils.load_json(self.paper_dir, 'clean-id2paper-test-{}.json'.format(fold))

    def get_candidates_by_ii(self, npaper, word2ids):
        title = npaper['title']
        words = data_utils.get_words(title, window=self.ii_window)
        cids = []
        for word in words:
            if word in word2ids:
                cids += word2ids[word]
        cids = list(set(cids))
        return cids


if __name__ == '__main__':
    paper_data_utils = PaperDataUtils()
    paper_data_utils.build_inverted_index(0)
