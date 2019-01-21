import argparse
import os
import tensorflow as tf
import numpy as np
import scipy.io as scio
from tfrbm.rbm import BernoulliRBM
from tfrbm.dataset import load_mnist
from tfrbm.utils import logit_mean, mnist_data
import matplotlib.pyplot as plt
from tfrbm.utils import np_sample_bernoulli
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

X, X_test = mnist_data(args.precision)

rbm = BernoulliRBM(n_visible=784, vb_init=logit_mean(X) if args.sparsity else None, n_hidden=args.n_hidden, precision=args.precision, algorithm=args.algorithm, anneal_lr=args.anneal_lr, n_gibbs_step=args.n_gibbs_steps, learning_rate=args.lr, \
                   use_momentum=args.momentum, momentum=[0.5,0.5,0.5,0.5,0.5,0.9], max_epoch=args.epochs, batch_size=args.batch_size, regularization=args.regularization, \
                   rl_coeff=1e-4, sample_h_state=True, sample_v_state=args.sample_v, save_path=args.save_path, save_after_each_epoch=False, sparsity=args.sparsity, \
                    sparsity_cost=1e-4, sparsity_target=0.1, sparsity_damping=0.9, verbose=args.verbose, dropout=args.dropout, dropconnect=args.dropconnect, img_shape=(28, 28))
# using previous paper's weights
# weights_file = '../data/mnist/mnistvh_CD25.mat'
# weights = scio.loadmat(weights_file)
rbm._set_weights(weights['vishid'], weights['visbiases'], weights['hidbiases'])
batch = X_test[:100,]

# logZ, avg_logp = rbm.fit(X, X_test)
# rbm._save_weights('original_sala')
if args.n_hidden < 30:
    logZ, avg_logp = eval_logp(rbm._sess, True, rbm._w, rbm._vb, rbm._hb, X, X_test, args.precision, 100, 100)
else:
    logZ, avg_logp = eval_logp(rbm._sess, False, rbm._w, rbm._vb, rbm._hb, X, X_test, args.precision, 100, 100)
print('logZ:%f,test average logp:%f' % (logZ, avg_logp))
rbm._sess.close()
