from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple
from utils.settings import *
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime
# from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from utils import eval_utils
from utils import data_utils


class Title2Vec:
    dim = 100
    train_data_dir = join(DATA_DIR, 'doc2vec')
    train_data_fname = 'doc2vec-train-corpus.json'
    model_dir = join(TRAIN_DIR, 'doc2vec')
    model_fname = 'doc2vec.mod'
    paper_dir = PAPER_DIR
    vectors_dir = join(OUT_DIR, 'doc2vec')
    threshold = 0.5
    sim_dir = join(OUT_DIR, 'sim')
    ranking_dir = join(OUT_DIR, 'ranking', 'doc2vec')

    def __init__(self):
        self.model = None
        pass

    def create_model(self):
        model = Doc2Vec(size=self.dim, min_count=1, dm=0, hs=1,
                    alpha=0.025, min_alpha=0.025, negative=0,
                    sample=0, dm_concat=1)  # parameter setting
        return model

    def load_model(self, force=False):
        if not self.model or force:
            print('loading model')
            self.model = Doc2Vec.load(join(self.model_dir, self.model_fname))
            print('model loaded')
        return self.model

    def prepare_corpus(self):
        train_corpus_analyzed = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        train_corpus = data_utils.load_json(self.train_data_dir, self.train_data_fname)
        print('training documents loaded')
        print('documents number: {}'.format(len(train_corpus)))
        for i, text in enumerate(train_corpus):
            if i % 10000 == 0:
                print(i)
            words = data_utils.get_words(text)
            tags = [i]
            train_corpus_analyzed.append(analyzedDocument(words=words, tags=tags))
            # if i > 100000:
            #     break
        return train_corpus_analyzed

    def train(self):
        start_time = datetime.now()
        model = self.create_model()
        train_corpus_analyzed = self.prepare_corpus()
        print('building vocabulary...')
        model.build_vocab(train_corpus_analyzed)
        print('building vocab finished')

        iter_num = 5  # large iteration number is harmful
        for epoch in range(iter_num):
            print('epoch', epoch)
            model.train(train_corpus_analyzed, total_examples=len(train_corpus_analyzed), epochs=1)
            model.alpha -= 0.002
            model.min_alpha = model.alpha

        model.save(join(self.model_dir, self.model_fname))
        print('saved')
        end_time = datetime.now()
        print('doc2vec training time', end_time - start_time)

    def model2vec(self, model, titles, d):  # doc2vec model
        m = len(titles)
        vectors = np.zeros((m, d), dtype=np.double)
        for i, title in enumerate(titles):
            words = data_utils.get_words(title)
            vec = model.infer_vector(words)
            vectors[i, :] = vec
        return vectors

    def titles2vec_no_dump(self, src_papers, dst_papers):
        model = self.load_model()
        src_titles = [sp['title'] for sp in src_papers]
        dst_titles = [dp['title'] for dp in dst_papers]
        src_vectors_test = self.model2vec(model, src_titles, self.dim)
        src_vectors_test = self.scale_matrix(src_vectors_test)
        dst_vectors_test = self.model2vec(model, dst_titles, self.dim)
        dst_vectors_test = self.scale_matrix(dst_vectors_test)
        return src_vectors_test, dst_vectors_test

    def titles2vec(self, role, fold):
        if role == 'test':
            model = self.load_model()
            fname = '{}-papers-{}-{}.dat'
            cpapers_fname = fname.format('clean', 'test', fold)
            cpapers = data_utils.load_json_lines(self.paper_dir, cpapers_fname)
            ctitles = [cpaper['title'] for cpaper in cpapers]
            npapers_fname = fname.format('noisy', 'test', fold)
            npapers = data_utils.load_json_lines(self.paper_dir, npapers_fname)
            ntitles = [npaper['title'] for npaper in npapers]
            src_vectors_test = self.model2vec(model, ctitles, self.dim)
            src_vectors_test = self.scale_matrix(src_vectors_test)
            dst_vectors_test = self.model2vec(model, ntitles, self.dim)
            dst_vectors_test = self.scale_matrix(dst_vectors_test)
            src_vectors_fname = '{}-titles-doc2vec-{}-{}.pkl'.format('src', role, fold)
            # dump_data(src_vectors_test, self.vectors_dir, src_vectors_fname)
            dst_vectors_fname = '{}-titles-doc2vec-{}-{}.pkl'.format('dst', role, fold)
            # dump_data(dst_vectors_test, self.vectors_dir, dst_vectors_fname)
            return src_vectors_test, dst_vectors_test

    def load_vectors(self, role, fold):
        src_vectors_fname = '{}-titles-doc2vec-{}-{}.pkl'.format('src', role, fold)
        src_vectors = data_utils.load_data(self.vectors_dir, src_vectors_fname)
        dst_vectors_fname = '{}-titles-doc2vec-{}-{}.pkl'.format('dst', role, fold)
        dst_vectors = data_utils.load_data(self.vectors_dir, dst_vectors_fname)
        print('loaded')
        return src_vectors, dst_vectors

    def scale_matrix(self, mat):
        mn = mat.mean(axis=1)
        mat_center = mat - mn[:, None]
        return normalize(mat_center)

    def evaluate_precision_at_topK(self):
        fold = 5
        topK = [1, 5, 10]
        acc = np.zeros((fold, len(topK)), dtype=np.double)
        for i in range(fold):
            sorted_indices = data_utils.load_data(self.ranking_dir, 'ranking-indices-{}-{}.pkl'.format('test', i))
            for t, k in enumerate(topK):
                acc[i, t] = eval_utils.prec_at_top(sorted_indices, k)
            print(acc[i])
        print(acc.mean(axis=0))

    def evaluate(self, role, fold):
        start_time = datetime.now()
        eval_dict = {}
        src_vectors, dst_vectors = self.titles2vec(role, fold)
        load_complete_time = datetime.now()
        similarities = cosine_similarity(src_vectors, dst_vectors)
        # dump_data(similarities, self.sim_dir, 'doc2vec-sim-{}-{}.pkl'.format(role, fold))
        cal_sim_time = datetime.now()
        print('calculate similarities time', cal_sim_time - load_complete_time)
        sorted_indices = np.argsort(similarities, axis=1)
        sorted_indices = np.fliplr(sorted_indices)
        # for i in range(len(sorted_indices)):
        #     print(similarities[i, sorted_indices[i, 0]], i, sorted_indices[i, 0])
        # dump_data(sorted_indices, self.ranking_dir, 'ranking-indices-{}-{}.pkl'.format(role, fold))
        sort_complete_time = datetime.now()
        pred_time = sort_complete_time - load_complete_time
        eval_dict['pred_time'] = pred_time
        print('sort time', sort_complete_time - cal_sim_time)
        topK = [1, 5, 10]
        precision = []
        for k in topK:
            precision.append(eval_utils.prec_at_top(sorted_indices, k))
        print(precision)
        end_time = datetime.now()
        eval_dict['test_time'] = end_time - start_time
        return eval_dict


if __name__ == '__main__':
    doc2vec = Title2Vec()
    doc2vec.train()
    # doc2vec.titles2vec('test', 0)
    # doc2vec.load_vectors('test', 0)
    # doc2vec.evaluate('test', 0)
    # doc2vec.evaluate_precision_at_topK()

