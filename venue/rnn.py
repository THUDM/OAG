# from preprocess.word2vec import gen_vectors
from utils import preprocess
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, BatchNormalization, concatenate, Lambda
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
import numpy as np
import os
from os.path import join
import codecs
from utils.settings import *
from sklearn import metrics

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 32
BATCH_SIZE = 32
NUM_EPOCH = 20
DROP_RATE = 0.2
NUM_DENSE = 2
MAX_SENTENCE_LENGTH = int()
vocab_size = int()


def cal_auc(y_true, prob_pred):
    return metrics.roc_auc_score(y_true, prob_pred)


def cal_f1(y_true, prob_pred):
    f1 = 0
    threshold = 0
    pred = prob_pred.copy()
    for thr in [j * 0.01 for j in range(100)]:
        for i in range(len(prob_pred)):
            pred[i] = prob_pred[i] > thr
        performance = metrics.precision_recall_fscore_support(y_true, pred, average='binary')
        if performance[2] > f1:
            precision = performance[0]
            recall = performance[1]
            f1 = performance[2]
            threshold = thr
    return threshold, precision, recall, f1


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        # self.val_recalls = []
        # self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation_data[6]
        val_predict = self.model.predict(
            [self.validation_data[0], self.validation_data[1], self.validation_data[2], self.validation_data[3],
             self.validation_data[4], self.validation_data[5]])
        threshold, precision, recall, f1 = cal_f1(val_targ, val_predict)
        print(
            "threshold:{:2f}\t precision:{:4f}\trecall:{:4f}\tf1score:{:4f}\n".format(threshold, precision, recall, f1))
        return


metric = Metrics()


def Minus(inputs):
    x, y = inputs
    return (x - y)


def load_data():
    global vocab_size, MAX_SENTENCE_LENGTH
    MAX_SENTENCE_LENGTH, vocab_size, labels, mag, aminer, rate, mag_len, aminer_len, omag, oaminer, reverse = preprocess.gen_vectors()
    rate = np.array([rate[i] * 32 for i in range(len(rate))])
    return train_test_split(mag, aminer, labels, mag_len, aminer_len, rate, omag, oaminer, reverse, test_size=0.3)


def build_model():
    embedding_layer_1 = Embedding(vocab_size + 1, EMBEDDING_SIZE, input_length=8)
    embedding_layer_2 = Embedding(vocab_size + 1, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH)
    lstm_layer1 = LSTM(HIDDEN_LAYER_SIZE, dropout=DROP_RATE, recurrent_dropout=DROP_RATE, return_sequences=True)
    lstm_layer2 = LSTM(HIDDEN_LAYER_SIZE, dropout=DROP_RATE, recurrent_dropout=DROP_RATE, return_sequences=False)

    lstm_layer3 = LSTM(HIDDEN_LAYER_SIZE, dropout=DROP_RATE, recurrent_dropout=DROP_RATE, return_sequences=True)
    lstm_layer4 = LSTM(HIDDEN_LAYER_SIZE, dropout=DROP_RATE, recurrent_dropout=DROP_RATE, return_sequences=False)

    seq_input_1 = Input(shape=(8,), dtype='int32')
    y1 = lstm_layer1(embedding_layer_1(seq_input_1))
    y1 = lstm_layer2(y1)

    seq_input_2 = Input(shape=(8,), dtype='int32')
    y2 = lstm_layer1(embedding_layer_1(seq_input_2))
    y2 = lstm_layer2(y2)

    seq_input_3 = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
    y3 = lstm_layer3(embedding_layer_2(seq_input_3))
    y3 = lstm_layer4(y3)

    seq_input_4 = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
    y4 = lstm_layer3(embedding_layer_2(seq_input_4))
    y4 = lstm_layer4(y4)

    rate = Input(shape=(32,), dtype='float32')
    reverse = Input(shape=(16,), dtype='float32')
    # mag_len = Input(shape=(1,), dtype='float32')
    # aminer_len = Input(shape=(1,), dtype='float32')

    minus = Lambda(Minus)

    y = BatchNormalization()(concatenate(
        [y1, y2, y3, y4, minus([y3, y4]), minus([y1, y2]), rate, reverse]))
    y = Dense(64)(y)
    y = Dense(16)(y)
    preds = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[seq_input_1, seq_input_2, rate, seq_input_3, seq_input_4, reverse], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=["accuracy"])
    model.summary()
    return model


def train_rnn():
    X1train, X1valid, X2train, X2valid, labels_train, labels_valid, X1lentrain, X1lenvalid, X2lentrain, X2lenvalid, ratetrain, ratevalid, X1otrain, X1ovalid, X2otrain, X2ovalid, Rtrain, Rvalid = load_data()
    print('--------------------', X1otrain)
    model = build_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    model_checkpoint = ModelCheckpoint(join(SAVED_MODEL, 'model.h5'), save_best_only=True, save_weights_only=False)
    history = model.fit([X1train, X2train, ratetrain, X1otrain, X2otrain, Rtrain], labels_train,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCH,
                        shuffle=True,
                        validation_data=(
                            [X1valid, X2valid, ratevalid, X1ovalid, X2ovalid, Rvalid], labels_valid),
                        callbacks=[metric, model_checkpoint])
    with codecs.open(join(SAVED_MODEL, 'history.txt'), 'w', 'utf-8') as f:
        f.write(str(history))
    print("rnn successfully trained...")


def predict(mag_list, aminer_list, rate, omag, oaminer, reverse):
    model = load_model(join(SAVED_MODEL, 'model.h5'))
    if not os.path.exists(join(DATA_DIR, 'result')):
        os.makedirs(join(DATA_DIR, 'result'))
    f = codecs.open(join(DATA_DIR, 'result/predict_result.txt'), 'w', 'utf-8')
    predicts = model.predict([mag_list, aminer_list, rate, omag, oaminer, reverse])
    f.write(str(list(predicts)))
    return predicts
