import os
import _pickle
from keras.preprocessing.text import text
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
import codecs
from utils.settings import *
from os.path import join
import numpy as np
# from model.tfidf import out

MAX_SEQUENCE_LENGTH = 17


def reverse_pair(seq):
    result = 0
    for i in range(len(seq)):
        for j in range(i, len(seq)):
            if seq[j] < i + 1:
                result -= 1
    return result


def word2vec(seq1, seq2):
    pass


def remove_stopword(seq, d):
    s = []
    for word in seq:
        if word not in d:
            s.append(word)
    return s


def jaccard_index(seq1, seq2, d):
    # sseq1 = remove_stopword(seq1, d)
    # sseq2 = remove_stopword(seq2, d)
    sseq1 = seq1
    sseq2 = seq2
    intersect = set(sseq1).intersection(sseq2)
    n = len(intersect)
    n1 = len(set(seq1).intersection(seq2))
    rate = n1 / (len(seq1) + len(seq2) - n1)
    dd = set(sseq1).union(sseq2).difference(intersect)
    s1 = remove_stopword(sseq1, dd)
    s2 = remove_stopword(sseq2, dd)
    lookup = {}
    ss1 = []
    for i, word in enumerate(s1):
        lookup[word] = i + 1
        ss1.append(i)
    try:
        ss2 = [lookup[word] for word in s2]
    except:
        print(sseq1)
        print(sseq2)
        print(list(intersect))
    result = reverse_pair(ss2)
    return len(seq1), len(seq2), ss1, ss2, rate, result


def gen_vectors():
    d = []
    with codecs.open(join(TRAIN_DATA, 'stoplist.txt'), 'r', 'utf-8') as f:
        for word in f.readlines():
            d.append(word[:-1])
    global MAX_SEQUENCE_LENGTH
    print("Fit tokenizer...")
    f = codecs.open(join(TRAIN_DATA, 'train_data.txt'), 'r', 'utf-8')
    datalist = eval(f.read())
    shuffle(datalist)
    print("Input data shuffled...")
    labels = []
    data_mag = []
    data_aminer = []
    rate = []
    for label, mag, aminer in datalist:
        labels.append(label)
        x = text.text_to_word_sequence(mag)
        y = text.text_to_word_sequence(aminer)
        MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, max(len(x), len(y)))
        data_mag.append(x)
        data_aminer.append(y)
    if os.path.exists(join(SAVED_MODEL, 'tokenizer')):
        f1 = open(join(SAVED_MODEL, 'tokenizer'), 'rb')
        tokenizer = _pickle.load(f1)
    else:
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(data_mag + data_aminer)
        print("Save tokenizer...")
        _pickle.dump(tokenizer, codecs.open(join(SAVED_MODEL, 'tokenizer'), 'wb'))
    # tokenized_mag = tokenizer.texts_to_sequences(data_mag)
    # tokenized_aminer = tokenizer.texts_to_sequences(data_aminer)
    mag = []
    aminer = []
    mag_len = []
    aminer_len = []
    reverse = []
    max_len = 0
    for i in range(len(data_mag)):
        x, y, a, b, c, e = jaccard_index(data_mag[i], data_aminer[i], d)
        mag.append(a)
        aminer.append(b)
        rate.append([c])
        mag_len.append([x])
        aminer_len.append([y])
        reverse.append([e] * 16)
        max_len = max(max(len(a), len(b)), max_len)
    if os.path.exists(join(SAVED_MODEL, 'tokenizer')):
        f1 = open(join(SAVED_MODEL, 'tokenizer'), 'rb')
        tokenizer = _pickle.load(f1)
    else:
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(data_mag + data_aminer)
        print("Save tokenizer...")
        _pickle.dump(tokenizer, codecs.open(join(SAVED_MODEL, 'tokenizer'), 'wb'))
    # rate = out()
    sseq_mag = pad_sequences(tokenizer.texts_to_sequences(data_mag), maxlen=MAX_SEQUENCE_LENGTH)
    sseq_aminer = pad_sequences(tokenizer.texts_to_sequences(data_aminer), maxlen=MAX_SEQUENCE_LENGTH)
    seq_mag = pad_sequences(mag, maxlen=8)
    seq_aminer = pad_sequences(aminer, maxlen=8)

    return MAX_SEQUENCE_LENGTH, len(tokenizer.word_index), labels, seq_mag, seq_aminer, rate, np.array(
        mag_len), np.array(aminer_len), sseq_mag, sseq_aminer, np.array(reverse)



def gen_predict_vectors(fname):
    global MAX_SEQUENCE_LENGTH
    print("Fit tokenizer...")
    f = codecs.open(join(TRAIN_DATA, '{}.txt'.format(fname)), 'r', 'utf-8')
    datalist = eval(f.read())
    data_mag = []
    data_aminer = []
    labels = []
    data_mag = []
    data_aminer = []
    rate = []
    for label, mag, aminer in datalist:
        labels.append(label)
        x = text.text_to_word_sequence(mag)
        y = text.text_to_word_sequence(aminer)
        MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, max(len(x), len(y)))
        data_mag.append(x)
        data_aminer.append(y)
    if os.path.exists(join(SAVED_MODEL, 'tokenizer')):
        f1 = open(join(SAVED_MODEL, 'tokenizer'), 'rb')
        tokenizer = _pickle.load(f1)
    else:
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(data_mag + data_aminer)
        print("Save tokenizer...")
        _pickle.dump(tokenizer, codecs.open(join(SAVED_MODEL, 'tokenizer'), 'wb'))
    # tokenized_mag = tokenizer.texts_to_sequences(data_mag)
    # tokenized_aminer = tokenizer.texts_to_sequences(data_aminer)
    mag = []
    aminer = []
    mag_len = []
    aminer_len = []
    reverse = []
    d = []
    with codecs.open(join(TRAIN_DATA, 'stoplist.txt'), 'r', 'utf-8') as f:
        for i in f.readlines():
            d.append(i[:-1])
    for i in range(len(data_mag)):
        x, y, a, b, c, e = jaccard_index(data_mag[i], data_aminer[i], d)
        mag.append(a)
        aminer.append(b)
        rate.append([c] * 32)
        mag_len.append([x])
        aminer_len.append([y])
        reverse.append([e] * 16)
    if os.path.exists(join(SAVED_MODEL, 'tokenizer')):
        f1 = open(join(SAVED_MODEL, 'tokenizer'), 'rb')
        tokenizer = _pickle.load(f1)
    else:
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(data_mag + data_aminer)
        print("Save tokenizer...")
        _pickle.dump(tokenizer, codecs.open(join(SAVED_MODEL, 'tokenizer'), 'wb'))
    # rate = out()
    sseq_mag = pad_sequences(tokenizer.texts_to_sequences(data_mag), maxlen=MAX_SEQUENCE_LENGTH)
    sseq_aminer = pad_sequences(tokenizer.texts_to_sequences(data_aminer), maxlen=MAX_SEQUENCE_LENGTH)
    seq_mag = pad_sequences(mag, maxlen=8)
    seq_aminer = pad_sequences(aminer, maxlen=8)

    return MAX_SEQUENCE_LENGTH, len(tokenizer.word_index), labels, seq_mag, seq_aminer, np.array(rate), np.array(
        mag_len), np.array(aminer_len), sseq_mag, sseq_aminer, np.array(reverse)


def raw_data_loader():
    f = codecs.open(join(TRAIN_DATA, 'train_data.txt'), 'r', 'utf-8')
    datalist = eval(f.read())
    shuffle(datalist)
    print("Input data shuffled...")
    labels = []
    data_mag = []
    data_aminer = []
    for label, mag, aminer in datalist:
        labels.append(label)
        # x = text.text_to_word_sequence(mag)
        # y = text.text_to_word_sequence(aminer)
        data_mag.append(mag)
        data_aminer.append(aminer)
    return data_mag, data_aminer, labels


def scale_matrix(matrix):
    matrix -= np.mean(matrix, axis=0)
    size = matrix.shape[0]
    return np.array(matrix).reshape(size, size, 1)


def encode_binary_codes(b):
    encoded_codes = ''.join('1' if x else '0' for x in b)
    v_hex = hex(int(encoded_codes, 2))
    # print(v_hex)
    return v_hex[2:]
