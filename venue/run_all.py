import os
import csv
import codecs
import sys
from venue.rnn import train_rnn, predict
from utils import preprocess
# from preprocess import load, word2vec
from utils.settings import *
from os.path import join
from bson import ObjectId
from random import random
import numpy as np


def flatten(result):
    for el in result:
        if hasattr(el, "__iter__") and not isinstance(el, float):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def run(mode):
    if mode == 'train':
        if not os.path.exists(join(SAVED_MODEL, 'tokenizer')):
            preprocess.gen_vectors()
        train_rnn()
    else:
        #load.gen_data()
        MAX_SENTENCE_LENGTH, vocab_size, labels, mag, aminer, rate, mag_len, aminer_len, omag, oaminer, reverse = preprocess.gen_predict_vectors('data')
        predict_result = predict(mag, aminer, rate, omag, oaminer, reverse)
        with codecs.open(join(TRAIN_DATA, 'data_id.txt'), 'r', 'utf-8') as f:
            idlist = eval(f.read())
        f1 = codecs.open(join(RESULT, 'result_name.txt'), 'w', 'utf-8')
        i = 0
        predict_result = np.array(predict_result).transpose()
        ft = codecs.open(join(RESULT, 'test.txt'), 'w', 'utf-8')
        for judge, id in zip(flatten(predict_result), idlist):
            mag_key, aminer_key = id
            if judge > 0.8:
                """
                obj = dst.find_one({'_id': mag_key})
                if not obj:
                    obj = dict()
                    obj['instances'] = []
                    obj['instances'].append(aminer_key)
                    obj['_id'] = mag_key
                    dst.insert(obj)
                else:
                    obj['instances'].append(aminer_key)
                    dst.update({'_id': mag_key}, obj)
                """
                i += 1
                print(i)
        print("successfully done!")


# run('predict')
run('train')
