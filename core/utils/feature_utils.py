import re
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize


def get_words(content, window=None, remove_stopwords=True):
    content = content.lower()
    r = re.compile(r'[a-z]+')
    words = re.findall(r, content)
    if remove_stopwords:
        stpwds = stopwords.words('english')
        words = [w for w in words if w not in stpwds]
    if window is not None:
        words = words[:window]
    return words

def name_equal(n1, n2):
    return 1 if n1 == n2 else -1

def scale_matrix(mat):
    mn = mat.mean(axis=1)
    mat_center = mat - mn[:, None]
    return normalize(mat_center)

def encode_binary_codes(b):
    encoded_codes = ''.join('1' if x else '0' for x in b)
    v_hex = ''
    block_length = 16
    for i in range(0, len(encoded_codes), block_length):
        cur_hex = hex(int(encoded_codes[i:i+block_length], 2))
        v_hex += cur_hex[2:]
    # v_hex = hex(int(encoded_codes, 2))
    return v_hex
