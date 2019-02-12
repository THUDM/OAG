from os.path import join
import os
from nltk.corpus import stopwords
import codecs
import json
import pickle
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with codecs.open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


def load_json_lines(fpath, fname):
    items = []
    with codecs.open(join(fpath, fname), 'r', encoding='utf-8') as rf:
        for line in rf:
            paper = json.loads(line)
            paper['title'] = paper['title'].lower()
            items.append(paper)
    return items


def get_words(content, window=None, remove_stopwords=True):
    import re
    content = content.lower()
    r = re.compile(r'[a-z]+')
    words = re.findall(r, content)
    if remove_stopwords:
        stpwds = stopwords.words('english')
        words = [w for w in words if w not in stpwds]
    if window is not None:
        words = words[:window]
    return words


def subname_equal(n1, n2):
    return 1 if n1 == n2 else -1


def remove_stopwords(word_list):
    stpwds = stopwords.words('english')
    words = [w for w in word_list if w not in stpwds]
    return words
