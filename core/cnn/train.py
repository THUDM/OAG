from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
from os.path import join
import os
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from core.cnn.models import CNNMatchModel
from core.cnn.data_loader import CNNMatchDataset
from core.utils.data_utils import ChunkSampler
from core.utils import feature_utils
from core.utils import eval_utils
from core.utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
parser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
parser.add_argument('--mat1-channel1', type=int, default=8, help='Matrix1 number of channels1.')
parser.add_argument('--mat1-kernel-size1', type=int, default=3, help='Matrix1 kernel size1.')
parser.add_argument('--mat1-channel2', type=int, default=16, help='Matrix1 number of channel2.')
parser.add_argument('--mat1-kernel-size2', type=int, default=2, help='Matrix1 kernel size2.')
parser.add_argument('--mat1-hidden', type=int, default=512, help='Matrix1 hidden dim.')
parser.add_argument('--mat2-channel1', type=int, default=8, help='Matrix2 number of channels1.')
parser.add_argument('--mat2-kernel-size1', type=int, default=2, help='Matrix2 kernel size1.')
parser.add_argument('--mat2-hidden', type=int, default=512, help='Matrix2 hidden dim')
parser.add_argument('--build-index-window', type=int, default=5, help='Matrix2 hidden dim')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--initial-accumulator-value', type=float, default=0.01, help='Initial accumulator value.')
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
parser.add_argument('--file-dir', type=str, default=settings.PAPER_DATA_DIR, help="Input file directory")
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


def train(epoch, train_loader, model, optimizer, args=args):
    model.train()

    loss = 0.
    total = 0.

    for i_batch, batch in enumerate(train_loader):
        X_title, X_author, Y = batch
        # print(Y)
        bs = Y.shape[0]

        if args.cuda:
            X_title = X_title.cuda()
            X_author = X_author.cuda()
            Y = Y.cuda()

        optimizer.zero_grad()
        output, _ = model(X_title.float(), X_author.float())

        loss_train = F.nll_loss(output, Y)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss epoch %d: %f", epoch, loss / total)


def evaluate(dataset, model, args=args):
    npapers_test = dataset.get_noisy_papers_test()
    id2cpaper = dataset.get_id2cpapers()
    word2ids = dataset.build_cpapers_inverted_index()
    preds = []
    labels = []
    n_papers = len(npapers_test)

    for i, npaper in enumerate(npapers_test):
        if i % 100 == 0:
            logger.info('npaper %d/%d', i, n_papers)
        if i > 0 and i % 1000 == 0:
            r = eval_utils.eval_prec_rec_f1_ir(preds, labels)
            logger.info('Testing %d samples: Prec: %.4f Rec: %.4f F1: %.4f', i, r[0], r[1], r[2])

        cids = dataset.get_candidates_by_inverted_index(npaper, word2ids)
        cpapers = [id2cpaper[cid] for cid in cids]
        pid = str(npaper['id'])
        if len(cpapers) == 0:
            preds.append(None)
            labels.append(pid)
            continue

        # predict
        cur_batch_matrix1 = []
        cur_batch_matrix2 = []
        for cpaper in cpapers:
            matrix1 = dataset.titles_to_matrix(cpaper['title'], npaper['title'])
            cur_batch_matrix1.append(feature_utils.scale_matrix(matrix1))
            matrix2 = dataset.authors_to_matrix(cpaper['authors'], npaper['authors'])
            cur_batch_matrix2.append(feature_utils.scale_matrix(matrix2))
        cur_batch_matrix1 = torch.FloatTensor(cur_batch_matrix1)
        cur_batch_matrix2 = torch.FloatTensor(cur_batch_matrix2)

        if args.cuda:
            cur_batch_matrix1 = cur_batch_matrix1.cuda()
            cur_batch_matrix2 = cur_batch_matrix2.cuda()

        _, y_pred = model(cur_batch_matrix1, cur_batch_matrix2)
        y_pred = [y[1] for y in y_pred]

        sorted_indices = sorted(range(len(y_pred)), key=lambda k: y_pred[k], reverse=True)  # in ypred
        pred_ids = [cids[i] for i in sorted_indices]
        preds.append(pred_ids[0] if len(pred_ids) > 0 and y_pred[sorted_indices[0]] > 0.5 else None)
        labels.append(pid)

    r = eval_utils.eval_prec_rec_f1_ir(preds, labels)
    logger.info('Testing Prec: %.4f Rec: %.4f F1: %.4f', r[0], r[1], r[2])


def main(args=args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    dataset_train = CNNMatchDataset(args.file_dir, args.matrix_size1, args.matrix_size2, args.build_index_window, args.seed, args.shuffle)
    N = len(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch,
                              sampler=ChunkSampler(N, 0))
    model = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                          mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                          mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                          mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                          mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
    model = model.float()

    if args.cuda:
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr,
                              initial_accumulator_value=args.initial_accumulator_value,
                              weight_decay=args.weight_decay)
    t_total = time.time()
    logger.info("training...")

    for epoch in range(args.epochs):
        train(epoch, train_loader, model, optimizer, args=args)

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    model_dir = join(settings.OUT_DIR, 'papers')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), join(model_dir, 'paper-matching-cnn.mdl'))
    logger.info('paper matching CNN model saved')

    evaluate(dataset_train, model, args)


if __name__ == '__main__':
    main(args=args)
