import numpy as np
import tensorflow as tf
import scipy.io as scio
from tqdm import tqdm
from .dataset import load_mnist

def logit_mean(X):
    p = np.mean(X, axis=0)
    p = np.clip(p, 1e-7, 1.-1e-7)
    q = np.log(p / (1. - p))
    q = q.reshape((1, len(q)))
    return q

def make_list_from(x):
    return list(x) if hasattr(x, '__iter__') else [x]
    
def assert_shape(obj, name, desired_shape):
    actual_shape = getattr(obj, name).shape
    if actual_shape != desired_shape:
        raise ValueError('`{0}` has invalid shape {1} != {2}'.\
                         format(name, actual_shape, desired_shape))

def assert_len(obj, name, desired_len):
    actual_len = len(getattr(obj, name))
    if actual_len != desired_len:
        raise ValueError('`{0}` has invalid len {1} != {2}'.\
                         format(name, actual_len, desired_len))

def np_sample_bernoulli(probs):
    return np.random.binomial(1, probs)

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs), dtype=probs.dtype)))

def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def batch_iter(X, batch_size, verbose, epoch):
    X = np.array(X)
    N = X.shape[0]
    n_batch = int(N / batch_size + (N % batch_size > 0))
    gen = range(n_batch)
    desc = 'epoch%d: ' % epoch
    if verbose: gen = tqdm(gen, leave=True, ncols=64, ascii=True, desc=desc)
    for i in gen:
        if i == n_batch - 1: yield X[i*batch_size:]
        else: yield X[i*batch_size:(i+1)*batch_size]

# This function loads the mnist test data from Salakhutdinov et al.'s RBM_AIS code.
def mnist_data(precision):
    print("\nPreparing data ...\n\n")
    X, _ = load_mnist(mode='train')
    X = X.astype(precision)
    X /= 255.

    testFile = '../data/mnist/testdata.mat'
    test_data = scio.loadmat(testFile)

    X_test_ = test_data['testbatchdata']
    X_test = np.zeros((10000, 784), dtype=precision)
    for i in range(100):
        for j in range(100):
            X_test[i*100+j] = X_test_[i,:,j]
    
    return X, X_test
