import os
from os.path import join
import numpy as np
from gensim.models import Doc2Vec
from collections import namedtuple
import time

from core.utils import data_utils
from core.utils import feature_utils
from core.utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class Title2Vec:
    model = None
    model_dir = join(settings.OUT_DIR, 'doc2vec')
    os.makedirs(model_dir, exist_ok=True)
    def __init__(self, dim=100):
        self.dim = dim
        self.model_fname = 'doc2vec-{}.mod'.format(self.dim)

    def load_model(self):
        logger.info('loading doc2vec model name %s', self.model_fname)
        self.model = Doc2Vec.load(join(self.model_dir, self.model_fname))
        logger.info('doc2vec model %s loaded', self.model_fname)
        return self.model

    def create_model(self):
        model = Doc2Vec(vector_size=self.dim, min_count=1, dm=0, hs=1,
                    alpha=0.025, min_alpha=0.025, negative=0,
                    sample=0, dm_concat=1)  # parameter setting
        return model

    def prepare_corpus(self):
        train_corpus_analyzed = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        train_corpus = data_utils.load_json(settings.PAPER_DATA_DIR, 'doc2vec-train-corpus.json')
        logger.info('training documents loaded')
        logger.info('documents number: {}'.format(len(train_corpus)))
        for i, text in enumerate(train_corpus):
            if i % 10000 == 0:
                logger.info('process doc %d', i)
            words = feature_utils.get_words(text)
            tags = [i]
            train_corpus_analyzed.append(analyzedDocument(words=words, tags=tags))
        return train_corpus_analyzed

    def train(self):
        start_time = time.time()
        model = self.create_model()
        train_corpus_analyzed = self.prepare_corpus()
        logger.info('building vocabulary...')
        model.build_vocab(train_corpus_analyzed)
        logger.info('building vocab finished')

        iter_num = 5
        for epoch in range(iter_num):
            logger.info('epoch %d', epoch)
            model.train(train_corpus_analyzed, total_examples=len(train_corpus_analyzed), epochs=1)
            model.alpha -= 0.002
            model.min_alpha = model.alpha

        model.save(join(self.model_dir, self.model_fname))
        logger.info('doc2vec model saved')
        end_time = time.time()
        logger.info('doc2vec training time %.4fs', end_time - start_time)

    def titles2vec(self, titles):  # doc2vec model
        if self.model is None:
            self.load_model()
        m = len(titles)
        vectors = np.zeros((m, self.dim), dtype=np.double)
        for i, title in enumerate(titles):
            words = feature_utils.get_words(title)
            vec = self.model.infer_vector(words)
            vectors[i, :] = vec
        return vectors

    def prepare_paper_title_to_vectors(self):
        if self.model is None:
            self.load_model()

        src_vectors_fname = '{}-titles-doc2vec-test.pkl'.format('src')
        dst_vectors_fname = '{}-titles-doc2vec.pkl'.format('dst')


        if os.path.isfile(join(settings.OUT_PAPER_DIR, src_vectors_fname)) \
            and os.path.isfile(join(settings.OUT_PAPER_DIR, dst_vectors_fname)):
            src_vectors_test = data_utils.load_large_obj(settings.OUT_PAPER_DIR, src_vectors_fname)
            dst_vectors = data_utils.load_large_obj(settings.OUT_PAPER_DIR, dst_vectors_fname)
            return src_vectors_test, dst_vectors


        fname = '{}-papers-{}.dat'
        cpapers_fname_train = fname.format('clean', 'train')
        cpapers_train = data_utils.load_json_lines(settings.PAPER_DATA_DIR, cpapers_fname_train)
        cpapers_fname_test = fname.format('clean', 'test')
        cpapers_test = data_utils.load_json_lines(settings.PAPER_DATA_DIR, cpapers_fname_test)
        cpapers = cpapers_train + cpapers_test
        ctitles = [cpaper['title'].lower() for cpaper in cpapers]

        npapers_fname = fname.format('noisy', 'test')
        npapers_test = data_utils.load_json_lines(settings.PAPER_DATA_DIR, npapers_fname)
        ntitles_test = [npaper['title'].lower() for npaper in npapers_test]

        src_vectors_test = self.titles2vec(ntitles_test)
        src_vectors_test = feature_utils.scale_matrix(src_vectors_test)  # useful

        dst_vectors = self.titles2vec(ctitles)
        dst_vectors = feature_utils.scale_matrix(dst_vectors)
        data_utils.dump_large_obj(src_vectors_test, settings.OUT_PAPER_DIR, src_vectors_fname)

        data_utils.dump_large_obj(dst_vectors, settings.OUT_PAPER_DIR, dst_vectors_fname)
        return src_vectors_test, dst_vectors


if __name__ == '__main__':
    title2vec = Title2Vec()
    title2vec.train()
    title2vec.prepare_paper_title_to_vectors()
    logger.info('done')
