import numpy as np


def prec_at_top_withid(preds, labels, k):
    n = len(preds)
    hits = np.zeros(n, dtype=int)
    for i in range(n):
        # print(labels[i])
        # print(preds[i][:k])
        hits[i] = int(labels[i] in preds[i][:k])
    return hits.mean()


def prec_at_top(pred_matrix, k):
    """
    :param pred_matrix: predict matrix, each element is an 'index', leftmost is the most confident
    :param k:
    :return:
    """
    m = pred_matrix.shape[0]
    count = 0
    for i in range(m):
        count += int(i in pred_matrix[i, :k])
    print(count / m)
    return count/m
