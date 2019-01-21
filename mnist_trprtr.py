import argparse
import os
import tensorflow as tf
import numpy as np
import scipy.io as scio
from tfrbm.rbm import BernoulliRBM
from tfrbm.dataset import load_mnist
from tfrbm.utils import tf_count
import matplotlib.pyplot as plt
from tfrbm.rbm_ais import eval_logp
from tfrbm.utils import np_sample_bernoulli, mnist_data

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

args = parser.parse_args()

if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)

X, X_test = mnist_data(args.precision)

rbm = BernoulliRBM(n_visible=784, n_hidden=args.n_hidden, precision=args.precision, algorithm=args.algorithm, anneal_lr=args.anneal_lr, n_gibbs_step=args.n_gibbs_steps, learning_rate=args.lr, \
                   use_momentum=args.momentum, momentum=[0.5,0.5,0.5,0.5,0.5,0.9], max_epoch=args.epochs, batch_size=args.batch_size, regularization=args.regularization, \
                   rl_coeff=1e-4, sample_h_state=True, sample_v_state=args.sample_v, save_path=args.save_path, save_after_each_epoch=False, sparsity=args.sparsity, \
                    sparsity_cost=1e-4, sparsity_target=0.1, sparsity_damping=0.9, verbose=args.verbose, img_shape=(28, 28))

# using previous paper's weights
weights_file = './data/mnist/mnistvh_CD25.mat'
weights = scio.loadmat(weights_file)
rbm._set_weights(weights['vishid'], weights['visbiases'], weights['hidbiases'])
rbm._save_weights('original_sala')

batch = X_test[:100,]

def train(rbm):
    logZ, avg_logp = rbm.fit(X, X_test)
    print('after training,logZ:%f,average_logp:%f' % (logZ, avg_logp))
    rbm.sample(batch, 100, 10000, 'ori')

# pruning in a single agreesive probability way without retraining
def pruning_woretrain(rbm, ckpt):
    print('\n\npruning in a single agreesive probability way without retraining\n\n')
    for sparsity in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rbm._load_weights(ckpt)
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, sparsity)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        rbm.reset_mask()

# pruning in a single agreesive probability way with retraining
def pruning_wretrain(rbm, ckpt):
    print('\n\npruning in a single agreesive probability way with retraining\n\n')
    for sparsity in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.95]:
        rbm._load_weights(ckpt)
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, sparsity)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        logZ, avg_logp = rbm.fit(X, X_test)
        print('after pruning and retraining,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        rbm.reset_mask()

# pruning in a single agreesive probability way with retraining and re-initialization
def pruning_wretrainreinit(rbm, ckpt):
    print('\n\npruning in a single agreesive probability way with retraining and re-initialization\n\n')
    for sparsity in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rbm._load_weights(ckpt)
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, sparsity)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        rbm.reinit_weight()
        rbm._w.assign(tf.multiply(rbm._w, rbm._mask))
        logZ, avg_logp = rbm.fit(X, X_test)
        print('after pruning and retraining,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        rbm.reset_mask()

# pruning in a single agreesive way by Song Han's method
def pruning_single_songhan(rbm, ckpt):
    print('\n\npruning in a single agreesive way by Song Han\'s method\n\n')
    parameters = [0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019]
    para = tf.placeholder(args.precision, ())
    threshold_update = para * tf.norm(rbm._w, 2)
    for i in range(len(parameters)):
        rbm._load_weights(ckpt)
        print('parameter:%f' % parameters[i])
        threshold = rbm._sess.run(threshold_update, feed_dict={para: parameters[i]})
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, 0, threshold)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        logZ, avg_logp = rbm.fit(X, X_test)
        print('after pruning and retraining,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        rbm.reset_mask()       

# pruning in a progressive probability way
def pruning_progressive_prob(rbm):
    print('\n\npruning in a progressive probability way with retraining\n\n')
    for sparsity in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, sparsity)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        logZ, avg_logp = rbm.fit(X, X_test)
        print('after pruning and retraining,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))

# iterative pruning and retraining without re-initialization by Song Han's method
def pruning_iter(rbm, iterations, para):
    print('\n\niterative pruning and retraining,without re-initialization-Song Han\'s method\n\n')
    threshold_update = para * tf.norm(rbm._w - tf.reduce_mean(rbm._w), 2)
    for i in range(iterations):
        print('\niteration-%d' % i)
        threshold = rbm._sess.run(threshold_update)
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, 0, threshold)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        logZ, avg_logp = rbm.fit(X, X_test, epoch=50)
        print('after pruning and retraining,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))


# iterative pruning and retraining in a probability way,without re-initialization
def pruning_iter_probability(rbm, iterations, prob):
    print('\n\niterative pruning and retraining in a probability way,without re-initialization\n\n')
    sparsity = 0.
    for i in range(iterations):
        print('\niteration-%d' % i)
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, sparsity + (1. - sparsity)* prob)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        logZ, avg_logp = rbm.fit(X, X_test, epoch=50)
        print('after pruning and retraining,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))    

def pruning_experiment():
    rbm._load_weights('original_sala')
    if args.n_hidden < 30:
        logZ, avg_logp = eval_logp(rbm._sess, True, rbm._w, rbm._vb, rbm._hb, X, X_test, args.precision, 100, 100)
    else:
        logZ, avg_logp = eval_logp(rbm._sess, False, rbm._w, rbm._vb, rbm._hb, X, X_test, args.precision, 100, 100)
    print('baseline:logZ:%f,test average_logp:%f' % (logZ, avg_logp)) 
    pruning_iter_probability(rbm, 7, 0.3)
    rbm._sess.close()

if __name__ == '__main__':
    pruning_experiment()

