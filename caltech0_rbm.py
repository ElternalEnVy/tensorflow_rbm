import argparse
import os
import tensorflow as tf
import numpy as np
import scipy.io as scio
from tfrbm.rbm import BernoulliRBM
import matplotlib.pyplot as plt
from tfrbm.rbm_ais import eval_logp

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--algorithm', type=str, default='CD', help = 'select training algorithm,CD or PCD')
parser.add_argument('--n-hidden', type=int, default=500, help='number of hidden units')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch-size', type=int, default=100, help='number of batch size')
parser.add_argument('--epochs', type=int, default=150, help = 'select training epochs')
parser.add_argument('--anneal-lr', default=False, action='store_true', help = 'use annealed learning')
parser.add_argument('--n-gibbs-steps', type=int ,default=1, help = 'num of gibbs steps for sampling')
parser.add_argument('--save-path', default='/data/experiments/model/rbm/mnist_rbm/', help = 'define saving path')
parser.add_argument('--precision', type=str, default='float32', help='data type precision')
parser.add_argument('--sparsity' , default=False, action='store_true', help='use sparsity penalty for hidden activaties or not')
parser.add_argument('--sample-v', default=True, action='store_false', help='not to sample V states')
parser.add_argument('--regularization', type=str, default='L2', help='use L1 or L2 regularization')
parser.add_argument('--momentum', default=True, action='store_false', help='not to use momentum')
parser.add_argument('--verbose', default=False, action='store_true', help='to use verbose display training process')
parser.add_argument('--dropout', type=float, default=None, help='dropout ratio')
parser.add_argument('--dropconnect', type=float, default=None, help='dropconnect ratio')

args = parser.parse_args()

if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)

data = scio.loadmat('/data/CalTech 101 Silhouettes Data Set/caltech101_silhouettes_28_split1.mat')
X = data['train_data']
X_val = data['val_data']
X = np.vstack((X, X_val))
X_test = data['test_data']
X = X.astype(args.precision)
X_test = X_test.astype(args.precision)
print(X.shape, X_test.shape)

batch = X_test[:100]

rbm = BernoulliRBM(n_visible=784, n_hidden=args.n_hidden, precision=args.precision, algorithm=args.algorithm, anneal_lr=args.anneal_lr, n_gibbs_step=args.n_gibbs_steps, learning_rate=args.lr, \
                   use_momentum=args.momentum, momentum=[0.5,0.5,0.5,0.5,0.5,0.9], max_epoch=args.epochs, batch_size=args.batch_size, regularization=args.regularization, \
                   rl_coeff=1e-4, sample_h_state=True, sample_v_state=args.sample_v, save_path=args.save_path, save_after_each_epoch=False, sparsity=args.sparsity, \
                    sparsity_cost=1e-4, sparsity_target=0.1, sparsity_damping=0.9, verbose=args.verbose, dropout=args.dropout, dropconnect=args.dropconnect, img_shape=(28, 28))

logZ, avg_logp = rbm.fit(X, X_test)
print('logZ:%f,test avg_logp:%f' % (logZ, avg_logp))
rbm._sess.close()
