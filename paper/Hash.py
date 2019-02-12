# coding = 'utf-8'

import numpy as np
from utils.settings import *
from datetime import datetime, timedelta
from collections import defaultdict as dd
from paper.Title2Vec import Title2Vec
from random import randint
from utils import data_utils
from utils import preprocess
from utils.eval_utils import prec_at_top

# pool = Pool(8)
other_bit = 26


def authors2b(authors):
    matrix = np.array((other_bit,))
    for author in authors:
        author = author.lower()
        name_words = data_utils.get_words(author)
        for name in name_words:
            bit_order = ord(name[0]) - ord('a')
            if 0 <= bit_order < other_bit:
                matrix[bit_order] = 1
    return matrix


class MHash:
    title_bit = 20
    other_bit = 26
    para_dir = join(DATA_DIR, 'hash')
    # distance_dir = join(OUT_DIR, 'distance')
    # ranking_dir = join(OUT_DIR, 'ranking')
    hash_table_dir = join(OUT_DIR, 'hash_table')
    paper_dir = join(DATA_DIR, 'papers')
    # threshold = 50
    threshold = 35
    # pool = Pool(8)

    def __init__(self, without_inner_results=False, with_all_attr=True):
        self.title2vec_model = Title2Vec()
        self.vectors_dim = self.title2vec_model.dim
        self.without_inner_results = without_inner_results
        self.with_all_attr = with_all_attr
        if self.with_all_attr:
            self.distance_dir = join(OUT_DIR, 'distance', 'all')
            self.ranking_dir = join(OUT_DIR, 'ranking', 'lsh', 'all')
        else:
            self.distance_dir = join(OUT_DIR, 'distance', 'titles')
            self.ranking_dir = join(OUT_DIR, 'ranking', 'lsh', 'titles')

    def load_title_vectors(self, role, fold):
        return self.title2vec_model.load_vectors(role=role, fold=fold)

    def prepare_LSH_parameters(self, role, fold):
        proj = np.random.normal(size=(self.vectors_dim, self.title_bit))
        fname = 'LSH_proj_matrix_{}_{}.pkl'.format(role, fold)
        if not self.without_inner_results:
            data_utils.dump_data(proj, self.para_dir, fname)
        return proj

    def load_LSH_parameters(self, role, fold):
        fname = 'LSH_proj_matrix_{}_{}.pkl'.format(role, fold)
        proj = data_utils.load_data(self.para_dir, fname)
        return proj

    def vectors2hash_LSH_micro(self, vectors, proj):
        proj_vectors = np.dot(vectors, proj)
        # B = np.zeros(proj_vectors.shape, dtype=bool)
        B = np.zeros(proj_vectors.shape)
        B = np.where(proj_vectors >= 0, B, 1)
        return B

    def vectors2hash_LSH_macro(self, role, fold):
        src_vectors, dst_vectors = self.load_title_vectors(role, fold)
        proj = self.load_LSH_parameters(role, fold)
        src_binary_codes = self.vectors2hash_LSH_micro(src_vectors, proj)
        dst_binary_codes = self.vectors2hash_LSH_micro(dst_vectors, proj)
        return src_binary_codes, dst_binary_codes

    def authors2binary_matrix(self, authors_list):
        m = len(authors_list)
        # matrix = np.zeros((m, self.other_bit), dtype=bool)
        matrix = np.zeros((m, self.other_bit))
        for i, authors in enumerate(authors_list):
            for author in authors:
                author = author.lower()
                name_words = data_utils.get_words(author)
                for name in name_words:
                    bit_order = ord(name[0]) - ord('a')
                    if 0 <= bit_order < self.other_bit:
                        matrix[i, bit_order] = 1
        # print(matrix)
        return matrix
        # return np.array(pool.map(authors2b, authors_list))

    def venues2binary_matrix(self, venues):
        import re
        m = len(venues)
        # matrix = np.zeros((m, self.other_bit), dtype=bool)
        matrix = np.zeros((m, self.other_bit))
        for i, venue in enumerate(venues):
            venue_words = re.split(' +', venue.strip())
            venue_words = data_utils.remove_stopwords(venue_words)
            # venue_words = get_words(venue, remove_stopwords=True)
            if 1 <= len(venue_words) <= 2:
                for c in venue_words[0].lower():
                    bit_order = ord(c) - ord('a')
                    if 0 <= bit_order < self.other_bit:
                        matrix[i, bit_order] = 1
            else:
                for word in venue_words:
                    if word:
                        chars_matter = ''
                        if word[0].isalpha():
                            chars_matter = word[0]
                        chars_matter += ''.join(c for c in word[1:] if c.isupper())
                        chars_matter = chars_matter.lower()
                        for c in chars_matter:
                            bit_order = ord(c) - ord('a')
                            if 0 <= bit_order < self.other_bit:
                                matrix[i, bit_order] = 1
        return matrix

    def years2binary_matrix(self, years):
        m = len(years)
        # matrix = np.zeros((m, self.other_bit), dtype=bool)
        matrix = np.zeros((m, self.other_bit))
        for i, year in enumerate(years):
            bit_order = min(self.other_bit, 2018 - year)
            for j in range(bit_order):
                matrix[i, j] = 1
        return matrix

    def build_binary2indices(self, binary_codes):
        m = binary_codes.shape[0]
        b2i_dict = dd(list)
        for i in range(m):
            encoded_codes = preprocess.encode_binary_codes(binary_codes[i])
            b2i_dict[encoded_codes].append(i)
        for k in b2i_dict:
            b2i_dict[k] = list(set(b2i_dict[k]))
        return b2i_dict

    def dump_hash_tables(self, role, fold):
        src_v, dst_v = self.vectors2hash_LSH_macro(role, fold)
        b2i_dict = self.build_binary2indices(dst_v)
        fname = '{}-title-hashtable-{}.json'.format(role, fold)
        data_utils.dump_json(b2i_dict, self.hash_table_dir, fname)

    def load_hash_tables(self, role, fold):
        fname = '{}-title-hashtable-{}.json'.format(role, fold)
        title_ht = data_utils.load_json(self.hash_table_dir, fname)
        return title_ht

    def cal_hamming_distance(self, B1, B2):
        # B1 = np.array(B1)
        # B2 = np.array(B2)
        # # print(B1)
        # # print(B2)
        # # print(B1.shape, B2.shape)
        # # print(B1.dtype, B2.dtype)
        # print(B1[:, None, :].shape)
        # r = (B1[:, None, :] != B2)
        # print('r', r)
        # return r.sum(2)
        r = 2 * np.inner(B1-0.5, 0.5-B2) + B1.shape[1]/2
        return r

    def construct_other_features(self, role, fold):
        fname = '{}-papers-{}-{}.dat'
        cpapers_fname = fname.format('clean', 'test', fold)
        cpapers = data_utils.load_json_lines(self.paper_dir, cpapers_fname)
        cauthors = [cpaper['authors'] for cpaper in cpapers]
        cvenues = [cpaper['venue'] for cpaper in cpapers]
        cyears = [cpaper['year'] for cpaper in cpapers]
        # print(cauthors)
        npapers_fname = fname.format('noisy', 'test', fold)
        npapers = data_utils.load_json_lines(self.paper_dir, npapers_fname)
        nauthors = [npaper['authors'] for npaper in npapers]
        nvenues = [npaper['venue'] for npaper in npapers]
        nyears = [npaper['year'] for npaper in npapers]

        src_authors_mat = self.authors2binary_matrix(cauthors)
        dst_authors_mat = self.authors2binary_matrix(nauthors)
        src_venues_mat = self.venues2binary_matrix(cvenues)
        dst_venues_mat = self.venues2binary_matrix(nvenues)
        src_years_mat = self.years2binary_matrix(cyears)
        dst_years_mat = self.years2binary_matrix(nyears)
        src_others_mat = np.concatenate((src_authors_mat, src_venues_mat, src_years_mat), axis=1)
        dst_others_mat = np.concatenate((dst_authors_mat, dst_venues_mat, dst_years_mat), axis=1)
        return src_others_mat, dst_others_mat

    def evaluate_prec_at_topk(self):
        fold = 5
        topK = [1, 5, 10]
        acc = np.zeros((fold, len(topK)), dtype=np.double)
        for i in range(fold):
            sorted_indices = data_utils.load_data(self.ranking_dir, 'ranking-indices-{}-{}.pkl'.format('test', i))
            for t, k in enumerate(topK):
                acc[i, t] = prec_at_top(sorted_indices, k)
            print(acc[i])
        print(acc.mean(axis=0))

    def evaluate(self, role, fold):
        start_time = datetime.now()
        eval_dict = {}
        src_title_vectors, dst_title_vectors = self.title2vec_model.titles2vec(role, fold)
        vec_time = datetime.now()
        print('convert to vectors', vec_time - start_time)
        # proj = self.load_LSH_parameters(role, fold)
        proj = self.prepare_LSH_parameters(role, fold)
        before_proj_time = datetime.now()
        src_binary_codes = self.vectors2hash_LSH_micro(src_title_vectors, proj)
        dst_binary_codes = self.vectors2hash_LSH_micro(dst_title_vectors, proj)
        after_proj_time = datetime.now()
        # print('proj time', after_proj_time - before_proj_time)

        if self.with_all_attr:
            src_other_mat, dst_other_mat = self.construct_other_features(role, fold)
            print('type', type(src_binary_codes))
            src_binary_codes = np.concatenate((src_binary_codes, src_other_mat), axis=1)
            dst_binary_codes = np.concatenate((dst_binary_codes, dst_other_mat), axis=1)
            build_other_features_time = datetime.now()
            print('build other features', build_other_features_time - after_proj_time)

        bc_time = datetime.now()
        print('vectors to binary codes', bc_time - vec_time)
        # src_binary_codes, dst_binary_codes = self.vectors2hash_LSH_macro(role, fold)
        distance_matrix = self.cal_hamming_distance(src_binary_codes, dst_binary_codes)
        # out_dir = join(self.distance_dir, 'title')
        fname = 'hamming_distance_{}_{}.pkl'.format(role, fold)
        # dump_data(distance_matrix, self.distance_dir, fname)
        before_dist_ranking_time = datetime.now()
        print('calculate hamming dist', before_dist_ranking_time - bc_time)
        distance_ranking_matrix = np.argsort(distance_matrix, axis=1)
        after_dist_ranking = datetime.now()
        print('ranking distance time', after_dist_ranking - before_dist_ranking_time)
        eval_dict['pred_time'] = after_dist_ranking - bc_time

        fname = 'ranking-indices-{}-{}.pkl'.format(role, fold)
        # dump_data(distance_ranking_matrix, self.ranking_dir, fname)
        # distance_ranking_matrix = load_data(self.ranking_dir, fname)
        # print(distance_ranking_matrix)
        self.prec_at_top(distance_ranking_matrix, 1)
        self.prec_at_top(distance_ranking_matrix, 5)
        self.prec_at_top(distance_ranking_matrix, 10)
        eval_dict['test_time'] = after_dist_ranking - start_time
        print(eval_dict['pred_time'])
        return eval_dict

    def evaluate_hashtables(self, role, fold):
        start_time = datetime.now()
        eval_dict = {}

        src_title_vectors, dst_title_vectors = self.title2vec_model.titles2vec(role, fold)
        # mv = src_title_vectors.shape[0]
        # vectors_same = 0
        # for i in range(mv):
        #     if np.array_equal(src_title_vectors[i], dst_title_vectors[i]):
        #         vectors_same += 1
        # proj = self.load_LSH_parameters(role, fold)
        proj = self.prepare_LSH_parameters(role, fold)
        # print(vectors_same/mv, vectors_same)

        before_proj_time = datetime.now()
        # dst
        dst_B = self.vectors2hash_LSH_micro(dst_title_vectors, proj)

        # src
        src_B = self.vectors2hash_LSH_micro(src_title_vectors, proj)
        after_proj_time = datetime.now()
        print('proj time', after_proj_time - before_proj_time)

        if self.with_all_attr:
            src_other_mat, dst_other_mat = self.construct_other_features(role, fold)
            src_B = np.concatenate((src_B, src_other_mat), axis=1)
            dst_B = np.concatenate((dst_B, dst_other_mat), axis=1)
            build_other_features_time = datetime.now()
            print('build other features', build_other_features_time - after_proj_time)
            # print(src_B.shape, dst_B.shape)

        before_build_dict = datetime.now()
        b2i_dict = self.build_binary2indices(dst_B)
        after_build_dict = datetime.now()
        # print('build b2i dict', after_build_dict - before_build_dict)

        m = src_B.shape[0]
        hit = 0
        hit_and_correct = 0
        for i in range(m):
            cur_v_hex = preprocess.encode_binary_codes(src_B[i])
            # print(cur_v_hex, encode_binary_codes(dst_B[i]))
            if cur_v_hex in b2i_dict:
                candidates = b2i_dict[cur_v_hex]
                rand_idx = randint(0, len(candidates) - 1)
                hit += 1
                if i == candidates[rand_idx]:
                    hit_and_correct += 1
        after_eval_time = datetime.now()
        eval_dict['pred_time'] = after_eval_time - after_build_dict
        print('look up table time', after_eval_time - after_build_dict)
        prec = hit_and_correct/hit
        rec = hit/m
        f1 = (2 * prec * rec)/(prec + rec)
        print('precision', prec, hit_and_correct, hit)
        print('recall', rec, m)
        print('f1', f1)
        predict_time = datetime.now() - start_time
        print('evaluation time', predict_time)
        eval_dict['test_time'] = predict_time
        return eval_dict
        # return prec, rec, f1

    def prec_at_top(self, pred_matrix, k):
        m = pred_matrix.shape[0]
        count = 0
        for i in range(m):
            count += int(i in pred_matrix[i, :k])
        print(count/m)


if __name__ == '__main__':
    mhash = MHash(with_all_attr=True)
    # mhash.prepare_LSH_parameters('test', 0)
    # mhash.vectors2hash_LSH('test', 0)
    # x = [True, False]
    # y = [False, True]
    # z = [True, True]
    # print(hamming(x, y))
    # print(hamming(x, z))
    # mhash.evaluate('test', 0)
    # mhash.dump_hash_tables('test', 0)
    # mhash.authors2binary_matrix([['jie tang', 'fanjin zhang'], ['jie tang', 'jing zhang']])
    # mhash.construct_other_features('test', 0)
    # mhash.evaluate_hashtables('test', 0)

    # x = np.array([[1,0,0], [1,0,1]])
    # x = np.zeros((9234, 260))
    # y = np.array([[1,0,1], [1,0,0]])
    # y = np.ones((9234, 260))
    # print(x)
    # print(mhash.cal_hamming_distance(x, y))
    # mhash.evaluate_prec_at_topk()
    print('done')
