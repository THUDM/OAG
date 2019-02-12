from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from author.models import MatchBatchGAT
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from author.data_loader import ChunkSampler
from author.data_loader import PairedLocalGraphDataset
from author.data_loader import processing_batch
from utils import settings

import os
from os.path import join
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--tensorboard-log', type=str, default='', help="name of this run")
parser.add_argument('--model', type=str, default='gat', help="models used")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="32,8",
                    help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8,1",
                    help="Heads in each layer, splitted with comma")
parser.add_argument('--batch', type=int, default=64, help="Batch size")
parser.add_argument('--dim', type=int, default=64, help="Embedding dimension")
parser.add_argument('--check-point', type=int, default=5, help="Eheck point")
parser.add_argument('--n-type-nodes', type=int, default=3, help="the number of different types of nodes")
parser.add_argument('--instance-normalization', action='store_true', default=True,
                    help="Enable instance normalization")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--save-features', action='store_true', default=True, help="Save hidden layer features")
parser.add_argument('--file-dir', type=str, default=join(settings.DATA_DIR, 'gat3'), help="Input file directory")
parser.add_argument('--alpha', type=float, default=0.2, help="Alpha for the leaky_relu.")
parser.add_argument('--opt', type=str, default='adam', help="Optimizer")
parser.add_argument('--tune-flag', type=bool, default=False, help="Tune flag")
parser.add_argument('--train-ratio', type=float, default=50, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=25, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=False,
                    help="Adjust weights inversely proportional"
                         " to class frequencies in the input data")
parser.add_argument('--use-vertex-feature', action='store_true', default=True,
                    help="Whether to use vertices' structural features")
parser.add_argument('--sequence-size', type=int, default=16,
                    help="Sequence size (only useful for pscn)")
parser.add_argument('--neighbor-size', type=int, default=5,
                    help="Neighborhood size (only useful for pscn)")

args = parser.parse_args()

if args.tune_flag:
    wf = open('results_tune_{}.csv'.format(args.file_dir), 'w')

dim_stat = 4


def evaluate(epoch, loader, model, loss_func, features_init, thr=None, return_best_thr=False, args=args, line_emb=None):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    svm_features_test = np.empty((0, dim_stat))
    sim_vec_test = np.empty((0, 8))
    labels_test = np.empty((0,))
    pred_scores = None
    attns = None
    v_types_all = None
    adjs_all = None
    labels_all = None
    for i_batch, batch in enumerate(loader):
        graph, labels, vertices, v_types_o, stat_features = batch

        features = torch.FloatTensor(features_init[vertices])
        if line_emb is not None:
            line_batch_embs = torch.FloatTensor(line_emb[vertices])
        else:
            line_batch_embs = None

        bs = len(labels)
        one_batch_nodes = len(graph) / bs
        n = vertices.shape[1]

        v_types = np.zeros((bs, n, args.n_type_nodes))
        v_types_o = v_types_o.numpy()
        for ii in range(bs):
            for vv in range(n):
                # idx = randint(0, args.n_type_nodes-1)
                idx = int(v_types_o[ii, vv])
                v_types[ii, vv, idx] = 1
                # v_types[ii, vv, 0] = 1
        v_types = torch.Tensor(v_types)  # bs x n x n_node_type

        if args.cuda:
            # features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            features = features.cuda()
            v_types = v_types.cuda()
            stat_features = stat_features.cuda()
            line_batch_embs = line_batch_embs.cuda()

        output, v_sim_mul, attn1 = model(features, graph, bs, one_batch_nodes, stat_features, line_batch_embs, v_types)
        # if args.model == "gcn" or args.model == "gat":
        #     output = output[:, -1, :]
        loss_batch = loss_func(output, labels)
        loss += bs * loss_batch.item()

        if attns is None:
            attns = attn1.cpu().detach().numpy()
            v_types_all = v_types_o
            adjs_all = graph.cpu().detach().numpy()
            labels_all = labels.cpu().detach().numpy()
        # else:
        #     attn1 = attn1.cpu().detach().numpy()
        #     attns = np.concatenate((attns, attn1), axis=0)
        #     v_types_all = np.concatenate((v_types_all, v_types_o), axis=0)
        #     cur_graph = graph.cpu().detach().numpy()
        #     adjs_all = np.concatenate((adjs_all, cur_graph), axis=0)

        y_true += labels.data.tolist()
        # y_true += labels.data.tolist() * 2 - 1
        # y_pred += output.max(1)[1].data.tolist()
        # y_pred += (output > 0.5).data.tolist()
        # y_pred += (output > 0).data.tolist()
        # y_score += output[:, 1].data.tolist()
        # y_score += output.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

        if args.save_features:
            svm_features_test = np.concatenate((svm_features_test, stat_features.cpu().detach().numpy()), axis=0)
            sim_vec_test = np.concatenate((sim_vec_test, v_sim_mul.cpu().detach().numpy()), axis=0)
            labels_test = np.concatenate((labels_test, labels.cpu().detach().numpy()), axis=0)

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1
        if args.save_features:
            pred_scores = np.array(y_pred)

    # logger.info('y_true %s', y_true)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                loss / total, auc, prec, rec, f1)
    if args.tune_flag:
        wf.write('auc, prec, rec, f1, {} {} {} {}\n'.format(auc, prec, rec, f1))

    out_dir = join(settings.OUT_DIR, 'svm-esb')
    os.makedirs(out_dir, exist_ok=True)

    if return_best_thr:
        mode = 'valid'
    else:
        mode = 'test'

    out_attn_dir = join(settings.OUT_DIR, 'attn')
    os.makedirs(out_attn_dir, exist_ok=True)
    np.save(join(out_attn_dir, 'attn_{}.npy'.format(mode)), attns)
    np.save(join(out_attn_dir, 'vtypes_{}.npy'.format(mode)), v_types_all)
    np.save(join(out_attn_dir, 'adjs_{}.npy'.format(mode)), adjs_all)
    np.save(join(out_attn_dir, 'labels_{}.npy'.format(mode)), labels_all)

    if return_best_thr:  # valid
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        if args.save_features:
            np.save(join(out_dir, 'stat_features_valid.npy'), svm_features_test)
            np.save(join(out_dir, 'structure_sim_vec_valid.npy'), sim_vec_test)
            np.save(join(out_dir, 'labels_valid.npy'), labels_test)

        return best_thr
    else:
        # np.save(join(args.file_dir, 'labels_test_gat.npy'), np.array(y_true).astype(np.dtype('B')))
        if args.save_features:
            np.save(join(out_dir, 'stat_features_test.npy'), svm_features_test)
            np.save(join(out_dir, 'structure_sim_vec_test.npy'), sim_vec_test)
            np.save(join(out_dir, 'labels_test.npy'), labels_test)
            np.save(join(out_dir, 'pred_scores_test.npy'), pred_scores)
        return None


def train(epoch, train_loader, valid_loader, test_loader, model, loss_func, optimizer, features_init, args=args,
          line_emb=None):
    model.train()

    loss = 0.
    total = 0.
    svm_features_train = np.empty((0, dim_stat))
    sim_vec_train = np.empty((0, 8))
    labels_train = np.empty((0,))
    attns = None
    v_types_all = None
    adjs_all = None
    labels_all = None
    for i_batch, batch in enumerate(train_loader):
        # graph, labels, features, v_types, svm_features = processing_batch(i_batch, batch, features_init)
        graph, labels, vertices, v_types_o, stat_features = batch

        features = torch.FloatTensor(features_init[vertices])
        if line_emb is not None:
            line_batch_embs = torch.FloatTensor(line_emb[vertices])
        else:
            line_batch_embs = None

        bs = len(labels)
        one_batch_nodes = len(graph) / bs
        n = vertices.shape[1]

        # v_types = [randBinList(vertices.shape[1]) for _ in range(vertices.shape[0])]
        v_types = np.zeros((bs, n, args.n_type_nodes))
        v_types_o = v_types_o.numpy()
        for ii in range(bs):
            for vv in range(n):
                # idx = randint(0, args.n_type_nodes-1)
                idx = int(v_types_o[ii, vv])
                v_types[ii, vv, idx] = 1
                # v_types[ii, vv, 0] = 1
        v_types = torch.Tensor(v_types)  # bs x n x n_node_type

        if args.cuda:
            # features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            features = features.cuda()
            v_types = v_types.cuda()
            stat_features = stat_features.cuda()
            line_batch_embs = line_batch_embs.cuda()

        optimizer.zero_grad()
        output, v_sim_mul, attn1 = model(features, graph, bs, one_batch_nodes, stat_features, line_batch_embs, v_types)
        # logger.info('output size %s', output.shape)
        # loss_train = F.nll_loss(output, labels, class_weight)

        if attns is None:
            attns = attn1.cpu().detach().numpy()
            v_types_all = v_types_o
            adjs_all = graph.cpu().detach().numpy()
            labels_all = labels.cpu().detach().numpy()
        # else:
        #     attn1 = attn1.cpu().detach().numpy()
        #     attns = np.concatenate((attns, attn1), axis=0)
        #     v_types_all = np.concatenate((v_types_all, v_types_o), axis=0)
        #     cur_graph = graph.cpu().detach().numpy()
        #     adjs_all = np.concatenate((adjs_all, cur_graph), axis=0)

        if args.save_features:
            svm_features_train = np.concatenate((svm_features_train, stat_features.cpu().detach().numpy()), axis=0)
            sim_vec_train = np.concatenate((sim_vec_train, v_sim_mul.cpu().detach().numpy()), axis=0)
            labels_train = np.concatenate((labels_train, labels.cpu().detach().numpy()), axis=0)

        # logger.info('output %s %s', output, labels)
        # loss_train = loss_func(output, labels)
        loss_train = F.nll_loss(output, labels)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss epoch %d: %f", epoch, loss / total)
    out_dir = join(settings.OUT_DIR, 'svm-esb')
    os.makedirs(out_dir, exist_ok=True)
    if args.save_features:
        np.save(join(out_dir, 'stat_features_train.npy'), svm_features_train)
        np.save(join(out_dir, 'structure_sim_vec_train.npy'), sim_vec_train)
        np.save(join(out_dir, 'labels_train.npy'), labels_train)

    if args.tune_flag:
        wf.write('train loss opoch {}: {}\n'.format(epoch, loss / total))
    # tensorboard_logger.log_value('train_loss', loss / total, epoch + 1)

    out_attn_dir = join(settings.OUT_DIR, 'attn')
    os.makedirs(out_attn_dir, exist_ok=True)
    np.save(join(out_attn_dir, 'attn_train.npy'), attns)
    np.save(join(out_attn_dir, 'vtypes_train.npy'), v_types_all)
    np.save(join(out_attn_dir, 'adjs_train.npy'), adjs_all)
    np.save(join(out_attn_dir, 'labels_train.npy'), labels_all)

    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint! validation...", epoch)
        if args.tune_flag:
            wf.write('eval on valid data ')
        best_thr = evaluate(epoch, valid_loader, model, loss_func, features_init, return_best_thr=True, args=args,
                            line_emb=line_emb)
        if args.tune_flag:
            wf.write('eval on test data ')
        logger.info('eval on test data!...')
        evaluate(epoch, test_loader, model, loss_func, features_init, thr=best_thr, args=args, line_emb=line_emb)


def main(args=args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('args cuda %s', args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    input_feature_dim = 200
    dataset = PairedLocalGraphDataset(
        args.file_dir, input_feature_dim, args.seed, args.shuffle)
    N = len(dataset)

    n_units = [input_feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")]
    print('init n_units', n_units)

    train_start, valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(valid_start - train_start, 0))
    valid_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(test_start - valid_start, valid_start))
    test_loader = DataLoader(dataset, batch_size=args.batch,
                             sampler=ChunkSampler(N - test_start, test_start))

    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = MatchBatchGAT(pretrained_emb=dataset.get_embedding(),
                          batch_size=args.batch,
                          vertex_feature=None,
                          use_vertex_feature=args.use_vertex_feature,
                          dim_pair_feature=dataset.dim_pair_features,
                          n_units=n_units, n_head=n_heads[0],
                          dropout=args.dropout,
                          instance_normalization=args.instance_normalization,
                          cuda=args.cuda,
                          alpha=args.alpha,
                          n_type_nodes=args.n_type_nodes,
                          attn_dropout=args.attn_dropout)
    features_init_all = dataset.get_embedding().numpy()
    line_emb_all = dataset.get_line_emb_mat()

    if args.cuda:
        model.cuda()
        logger.info('put model on gpu')

    logger.info('model paras %s', model)
    # optimizer = optim.Adagrad(model.layer_stack.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    # loss_func = torch.nn.BCELoss()
    loss_func = F.nll_loss

    # randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]

    # Train model
    t_total = time.time()
    logger.info("training...")
    for epoch in range(args.epochs):
        train(epoch, train_loader, valid_loader, test_loader, model, loss_func, optimizer, features_init_all, args=args,
              line_emb=line_emb_all)
        if args.tune_flag:
            wf.flush()
    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    logger.info("retrieve best threshold...")
    best_thr = evaluate(args.epochs, valid_loader, model, loss_func, features_init_all, return_best_thr=True, args=args,
                        line_emb=line_emb_all)

    # Testing
    logger.info("testing...")
    evaluate(args.epochs, test_loader, model, loss_func, features_init_all, thr=best_thr, args=args,
             line_emb=line_emb_all)
    model_dir = join(settings.OUT_DIR, 'gat-model')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), join(model_dir, 'hgat.mdl'))
    logger.info('gat model saved')


if __name__ == '__main__':
    main(args=args)
