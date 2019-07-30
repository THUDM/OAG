from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from os.path import join
import os
import time
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from core.gat.data_loader import PairedSubgraphDataset
from core.gat.models import MatchBatchHGAT
from core.utils.data_utils import ChunkSampler
from core.utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='hgat', help="models used")
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
parser.add_argument('--check-point', type=int, default=5, help="Check point")
parser.add_argument('--n-type-nodes', type=int, default=3, help="the number of different types of nodes")
parser.add_argument('--instance-normalization', action='store_true', default=True,
                    help="Enable instance normalization")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--file-dir', type=str, default=settings.AUTHOR_DATA_DIR, help="Input file directory")
parser.add_argument('--alpha', type=float, default=0.2, help="Alpha for the leaky_relu.")
parser.add_argument('--train-ratio', type=float, default=50, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=25, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=False,
                    help="Adjust weights inversely proportional"
                         " to class frequencies in the input data")
parser.add_argument('--use-vertex-feature', action='store_true', default=False,
                    help="Whether to use vertices' structural features")
parser.add_argument('--sequence-size', type=int, default=16,
                    help="Sequence size (only useful for pscn)")
parser.add_argument('--neighbor-size', type=int, default=5,
                    help="Neighborhood size (only useful for pscn)")

args = parser.parse_args()


def evaluate(epoch, loader, model, node_init_features, thr=None, return_best_thr=False, args=args):
    model.eval()
    total = 0.
    loss = 0.
    y_true, y_pred, y_score = [], [], []

    for i_batch, batch in enumerate(loader):
        graph, labels, vertices, v_types_orig = batch

        features = torch.FloatTensor(node_init_features[vertices])

        bs = len(labels)
        n = vertices.shape[1]

        v_types = np.zeros((bs, n, args.n_type_nodes))
        v_types_orig = v_types_orig.numpy()
        for ii in range(bs):
            for vv in range(n):
                idx = int(v_types_orig[ii, vv])
                v_types[ii, vv, idx] = 1
        v_types = torch.Tensor(v_types)  # bs x n x n_node_type

        if args.cuda:
            graph = graph.cuda()
            labels = labels.cuda()
            features = features.cuda()
            v_types = v_types.cuda()

        output = model(features, graph, v_types)
        loss_batch = F.nll_loss(output, labels)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                loss / total, auc, prec, rec, f1)

    if return_best_thr:  # valid
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None


def train(epoch, train_loader, valid_loader, test_loader, model, optimizer, node_init_features, args=args):
    model.train()

    loss = 0.
    total = 0.

    for i_batch, batch in enumerate(train_loader):
        graph, labels, vertices, v_types_orig = batch
        features = torch.FloatTensor(node_init_features[vertices])
        bs = len(labels)
        n = vertices.shape[1]

        v_types = np.zeros((bs, n, args.n_type_nodes))
        v_types_orig = v_types_orig.numpy()
        for ii in range(bs):
            for vv in range(n):
                idx = int(v_types_orig[ii, vv])
                v_types[ii, vv, idx] = 1
        v_types = torch.Tensor(v_types)  # bs x n x n_node_type

        if args.cuda:
            graph = graph.cuda()
            labels = labels.cuda()
            features = features.cuda()
            v_types = v_types.cuda()

        optimizer.zero_grad()
        output = model(features, graph, v_types)

        loss_train = F.nll_loss(output, labels)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()

    logger.info("train loss epoch %d: %f", epoch, loss / total)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint! validation...", epoch)
        best_thr = evaluate(epoch, valid_loader, model, node_init_features, return_best_thr=True, args=args)
        logger.info('eval on test data!...')
        evaluate(epoch, test_loader, model, node_init_features, thr=best_thr, args=args)


def main(args=args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    dataset = PairedSubgraphDataset(args.file_dir, args.seed, args.shuffle)
    N = len(dataset)

    input_feature_dim = dataset.get_node_input_feature_dim()
    n_units = [input_feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")]

    train_start, valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(valid_start - train_start, 0))
    valid_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(test_start - valid_start, valid_start))
    test_loader = DataLoader(dataset, batch_size=args.batch,
                             sampler=ChunkSampler(N - test_start, test_start))

    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = MatchBatchHGAT(n_type_nodes=args.n_type_nodes,
                           n_units=n_units,
                           n_head=n_heads[0],
                           dropout=args.dropout,
                           attn_dropout=args.attn_dropout,
                           instance_normalization=args.instance_normalization)
    node_init_features = dataset.get_embedding().numpy()

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    t_total = time.time()
    logger.info("training...")
    for epoch in range(args.epochs):
        train(epoch, train_loader, valid_loader, test_loader, model, optimizer, node_init_features, args=args)

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    logger.info("retrieve best threshold...")
    best_thr = evaluate(args.epochs, valid_loader, model, node_init_features, return_best_thr=True, args=args)

    # Testing
    logger.info("testing...")
    evaluate(args.epochs, test_loader, model, node_init_features, thr=best_thr, args=args)
    model_dir = join(settings.OUT_DIR, 'hgat-model')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), join(model_dir, 'hgat.mdl'))
    logger.info('gat model saved')


if __name__ == '__main__':
    main(args=args)
