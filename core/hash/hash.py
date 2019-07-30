from os.path import join
import os
import numpy as np
import time
from collections import defaultdict as dd

from core.hash.title2vec import Title2Vec
from core.utils import feature_utils
from core.utils import data_utils
from core.utils import eval_utils
from core.utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class HashMatch:
    title_bit = 128
    title2vec_model = Title2Vec()
    vector_dim = title2vec_model.dim
    proj = None

    def prepare_LSH_projection_matrix(self):
        proj = np.random.normal(size=(self.vector_dim, self.title_bit))
        fname = 'LSH_proj_matrix.pkl'
        data_utils.dump_large_obj(proj, settings.PAPER_DATA_DIR, fname)
        self.proj = proj

    def load_LSH_projection_matrix(self):
        fname = 'LSH_proj_matrix.pkl'
        proj = data_utils.load_large_obj(settings.PAPER_DATA_DIR, fname)
        self.proj = proj

    def title_vectors_to_binary_codes_LSH(self, vectors, proj):
        proj_vectors = np.dot(vectors, proj)
        B = np.zeros(proj_vectors.shape, dtype=np.bool_)
        B = np.where(proj_vectors >= 0, B, 1)
        return B

    def two_domain_title_vectors_to_binary_codes(self):
        src_vectors, dst_vectors = self.title2vec_model.prepare_paper_title_to_vectors()
        if self.proj is None:
            self.load_LSH_projection_matrix()
        src_binary_codes = self.title_vectors_to_binary_codes_LSH(src_vectors, self.proj)
        dst_binary_codes = self.title_vectors_to_binary_codes_LSH(dst_vectors, self.proj)
        return src_binary_codes, dst_binary_codes

    def dump_dst_hash_tables(self):
        src_binary_codes_test, dst_binary_codes = self.two_domain_title_vectors_to_binary_codes()
        hash_to_dst_idx = dd(list)
        cpapers_train = data_utils.load_json_lines(settings.PAPER_DATA_DIR, 'clean-papers-train.dat')
        cpapers_test = data_utils.load_json_lines(settings.PAPER_DATA_DIR, 'clean-papers-test.dat')
        cpapers = cpapers_train + cpapers_test
        for i, h in enumerate(dst_binary_codes):
            h = feature_utils.encode_binary_codes(h)
            hash_to_dst_idx[h].append(str(cpapers[i]['id']))
        data_utils.dump_json(hash_to_dst_idx, settings.OUT_PAPER_DIR, 'hash_to_dst_paper_id.json')

    def eval_hash_table(self):
        start_test_time = time.time()
        src_binary_codes_test, dst_binary_codes = self.two_domain_title_vectors_to_binary_codes()
        npapers_test = data_utils.load_json_lines(settings.PAPER_DATA_DIR, 'noisy-papers-test.dat')
        labels = [str(item['id']) for item in npapers_test]
        hash_to_dst_idx = data_utils.load_json(settings.OUT_PAPER_DIR, 'hash_to_dst_paper_id.json')
        preds = []
        before_loop_time = time.time()
        for i, h in enumerate(src_binary_codes_test):
            h = feature_utils.encode_binary_codes(h)
            if h in hash_to_dst_idx and len(hash_to_dst_idx[h]) == 1:
                preds.append(hash_to_dst_idx[h][0])
            else:
                preds.append(None)
        end_time = time.time()
        pred_time = end_time - before_loop_time
        test_time = end_time - start_test_time
        r = eval_utils.eval_prec_rec_f1_ir(preds, labels)
        logger.info('eval results: Prec. %.4f, Rec. %.4f, F1. %.4f', r[0], r[1], r[2])
        logger.info('test time %.2fs, predict time %.2fs', test_time, pred_time)


if __name__ == '__main__':
    hash_match = HashMatch()
    hash_match.prepare_LSH_projection_matrix()
    hash_match.dump_dst_hash_tables()
    hash_match.eval_hash_table()
    logger.info('done')
