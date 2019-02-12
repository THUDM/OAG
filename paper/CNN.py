from os.path import join
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.metrics import Accuracy
from tflearn.layers.conv import conv_2d
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.merge_ops import merge
from tflearn.data_utils import to_categorical
from sklearn.model_selection import train_test_split
from paper.paper_dataset import PaperDataUtils
from utils import preprocess
from utils import data_utils
from utils import eval_utils
from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp


class MCNNModel:
    paper_data_utils = PaperDataUtils()
    matrices_dir = join(settings.DATA_DIR, 'network-input')
    model_dir = join(settings.TRAIN_DIR, 'model')

    def __init__(self, add_author=True, title_mat_size=7, author_mat_size=4, ii_window=5):
        self.add_author = add_author
        self.title_mat_size = title_mat_size
        self.author_mat_size = author_mat_size
        self.ii_window = ii_window

    def titles2matrix(self, title1, title2):
        twords1 = data_utils.get_words(title1, remove_stopwords=False)[: self.title_mat_size]
        twords2 = data_utils.get_words(title2, remove_stopwords=False)[: self.title_mat_size]

        matrix = -np.ones((self.title_mat_size, self.title_mat_size))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

    def authors2matrix(self, authors1, authors2):
        matrix = -np.ones((self.author_mat_size, self.author_mat_size))
        author_num = int(self.author_mat_size/2)
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
                matrix[row][col] = data_utils.subname_equal(first_name1, first_name2)
                matrix[row][col+1] = data_utils.subname_equal(first_name1, last_name2)
                matrix[row+1][col] = data_utils.subname_equal(last_name1, first_name2)
                matrix[row+1][col+1] = data_utils.subname_equal(last_name1, last_name2)
        except Exception as e:
            # print('---', matrix)
            return matrix
        # print(matrix)
        return matrix

    def pairs2multiple_matrices(self, pairs):
        title_matrices = []
        author_matrices = []
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                logger.info('pairs to matrices %d', i)
            cpaper, npaper = pair
            matrix1 = self.titles2matrix(cpaper['title'], npaper['title'])
            title_matrices.append(matrix1)
            matrix2 = self.authors2matrix(cpaper['authors'], npaper['authors'])
            author_matrices.append(matrix2)
        return title_matrices, author_matrices

    def prepare_network_input(self, role, fold):
        """
        prepare cnn model input
        :param role: 'train' or 'test'
        :param fold: cross validation fold
        :return: constructed matrices
        """
        pos_pairs = self.paper_data_utils.construct_positive_paper_pairs(role, fold)
        logger.info('positive paper pairs built')
        neg_pairs = self.paper_data_utils.load_train_neg_paper_pairs(fold)
        logger.info('negative paper pairs loaded')
        pos_title_matrices, pos_author_matrices = self.pairs2multiple_matrices(pos_pairs)
        data_utils.dump_data(pos_author_matrices, self.matrices_dir, 'pos_author_matrices_{}.pkl'.format(fold))
        data_utils.dump_data(pos_title_matrices, self.matrices_dir, 'pos_title_matrices_{}.pkl'.format(fold))
        neg_title_matrices, neg_author_matrices = self.pairs2multiple_matrices(neg_pairs)
        data_utils.dump_data(neg_author_matrices, self.matrices_dir, 'neg_author_matrices_{}.pkl'.format(fold))
        data_utils.dump_data(neg_title_matrices, self.matrices_dir, 'neg_title_matrices_{}.pkl'.format(fold))
        return pos_title_matrices, pos_author_matrices, neg_title_matrices, neg_author_matrices

    def load_network_input(self, role, fold):  # load cnn model input
        if role == 'train':
            pos_title_matrices, pos_author_matrices, neg_title_matrices, neg_author_matrices = self.prepare_network_input(role, fold)

            n_positive_pairs = len(pos_title_matrices)
            n_negative_pairs = len(neg_title_matrices)
            n_matrix = n_positive_pairs + n_negative_pairs
            X_title = np.zeros((n_matrix * 2, self.title_mat_size, self.title_mat_size, 1), dtype='float')
            X_author = np.zeros((n_matrix * 2, self.author_mat_size, self.author_mat_size, 1), dtype='float')
            Y = np.zeros(n_matrix * 2)

            count = 0

            for i in range(n_positive_pairs):
                # if count % 100 == 0:
                #     print(count)
                matrix1 = pos_title_matrices[i]
                X_title[count] = preprocess.scale_matrix(matrix1)
                matrix2 = pos_author_matrices[i]
                X_author[count] = preprocess.scale_matrix(matrix2)
                Y[count] = 1
                count += 1

                X_title[count] = preprocess.scale_matrix(matrix1.transpose())
                X_author[count] = preprocess.scale_matrix(matrix2.transpose())
                Y[count] = 1
                count += 1

            del pos_title_matrices, pos_author_matrices

            for i in range(n_negative_pairs):
                # if count % 100 == 0:
                #     print(count)
                matrix1 = neg_title_matrices[i]
                X_title[count] = preprocess.scale_matrix(matrix1)
                matrix2 = neg_author_matrices[i]
                X_author[count] = preprocess.scale_matrix(matrix2)
                Y[count] = 0
                count += 1

                X_title[count] = preprocess.scale_matrix(matrix1.transpose())
                X_author[count] = preprocess.scale_matrix(matrix2.transpose())
                Y[count] = 0
                count += 1

            del neg_title_matrices, neg_author_matrices

            X_title_train, X_title_val, X_author_train, X_author_val, Y_train, Y_val = train_test_split(X_title,
                                                                                                        X_author, Y,
                                                                                                        test_size=0.1,
                                                                                                        random_state=42)
            Y_train = to_categorical(Y_train, 2)
            Y_validation = to_categorical(Y_val, 2)

            return X_title_train, X_title_val, X_author_train, X_author_val, Y_train, Y_validation

    def create_model_for_multiple_input(self):  # main cnn model
        network1 = input_data(shape=[None, self.title_mat_size, self.title_mat_size, 1])
        network2 = input_data(shape=[None, self.author_mat_size, self.author_mat_size, 1])

        activation = 'relu'
        # activation = 'tanh'
        # activation = 'sigmoid'

        network1 = conv_2d(network1, 8, 3, activation=activation, regularizer="L2")
        # network = max_pool_2d(network, 2)
        # network = local_response_normalization(network)
        network1 = conv_2d(network1, 16, 2, activation=activation, regularizer="L2")
        # network = max_pool_2d(network, 2)
        # network = local_response_normalization(network)
        network1 = fully_connected(network1, 512, activation=activation)

        network2 = conv_2d(network2, 8, 2, activation=activation, regularizer='L2')
        network2 = fully_connected(network2, 512, activation=activation)

        network = merge([network1, network2], 'concat')

        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        acc = Accuracy(name="Accuracy")

        adagrad = tflearn.AdaGrad(
            learning_rate=0.002, initial_accumulator_value=0.01)
        network = regression(network, optimizer=adagrad,
                              loss='categorical_crossentropy',
                              learning_rate=0.002, metric=acc)

        tmp_dir = join(settings.TRAIN_DIR, 'tmp')
        # util.mkdir(tmp_dir)
        model = tflearn.DNN(network,
                            checkpoint_path=join(tmp_dir, 'paper_similarity'),
                            max_checkpoints=10, tensorboard_verbose=3,
                            tensorboard_dir=join(tmp_dir, 'tflearn_logs'))
        return model

    def predict_similarities(self, npaper, cpapers, model):
        """

        :param npaper: one source paper
        :param cpapers: some candidate papers
        :param model: pre-trained cnn model
        :return: similarity scores
        """
        X = []
        X_title = []
        X_author = []
        for cpaper in cpapers:
            matrix1 = self.titles2matrix(cpaper['title'], npaper['title'])
            matrix1 = preprocess.scale_matrix(matrix1)
            X_title.append(matrix1)
            matrix2 = self.authors2matrix(cpaper['authors'], npaper['authors'])
            matrix2 = preprocess.scale_matrix(matrix2)
            X_author.append(matrix2)
            X = [X_title, X_author]
        try:
            ypred = model.predict(X)
        except:
            return []
        ypred = [y[1] for y in ypred]
        return ypred

    def train(self, fold):
        tf.reset_default_graph()  # restart kernel
        X_title, X_title_val, X_author, X_author_val, Y, Y_val = self.load_network_input('train', fold)
        model = self.create_model_for_multiple_input()
        model.fit([X_title, X_author], Y, validation_set=([X_title_val, X_author_val], Y_val),
                  batch_size=100, n_epoch=10, run_id=str(fold), show_metric=True)
        outpath = join(self.model_dir, 'cnn_model_{}.mod'.format(fold))
        model.save(outpath)

    def evaluate(self, fold):
        tf.reset_default_graph()  # restart kernel  # also need
        model = self.create_model_for_multiple_input()
        model.load(join(self.model_dir, 'cnn_model_{}.mod'.format(fold)))
        labels = []
        preds = []
        word2ids = self.paper_data_utils.load_inverted_index(fold)
        # word2ids = self.build_inverted_index(fold)
        npapers_fname = 'noisy-papers-test-{}.dat'.format(fold)
        npapers = data_utils.load_json_lines(settings.PAPER_DIR, npapers_fname)
        # id2cpaper = load_json(self.paper_dir, 'clean-id2paper-test-{}.json'.format(fold))
        id2cpaper = self.paper_data_utils.load_id2papers(fold)
        # possible_match = 0
        count = 0
        max_scores = []
        for npaper in npapers:
            count += 1
            if count % 10 == 0:
                print(count)
            pid = str(npaper['id'])
            # print('pid', pid)
            cids = self.paper_data_utils.get_candidates_by_ii(npaper, word2ids)
            # if pid in cids:
            #     possible_match += 1
            #     print('possible match', possible_match, 'passed', count)
            cpapers = [id2cpaper[cid] for cid in cids]
            ypred = self.predict_similarities(npaper, cpapers, model)
            sorted_indices = sorted(range(len(ypred)), key=lambda k: ypred[k], reverse=True)  # in ypred
            pred_ids = [cids[i] for i in sorted_indices]
            preds.append(pred_ids)
            if ypred:
                max_scores.append(ypred[sorted_indices[0]])
            else:
                max_scores.append(None)
            labels.append(pid)

        prec1 = eval_utils.prec_at_top_withid(preds=preds, labels=labels, k=1)
        prec5 = eval_utils.prec_at_top_withid(preds=preds, labels=labels, k=5)
        prec10 = eval_utils.prec_at_top_withid(preds=preds, labels=labels, k=10)
        print(prec1, prec5, prec10)


if __name__ == '__main__':
    mcnn = MCNNModel()
    mcnn.train(0)
    mcnn.evaluate(0)
