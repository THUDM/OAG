
def eval_prec_rec_f1_ir(preds, labels):
    true_map = 0
    mapped = 0
    m = len(preds)
    for i, pred_id in enumerate(preds):
        label = labels[i]  # str
        if pred_id is not None:  # str
            mapped += 1
            if pred_id == label:
                true_map += 1
    precision = true_map / mapped
    recall = true_map / m
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1
