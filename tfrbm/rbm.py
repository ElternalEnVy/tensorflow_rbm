import tensorflow as tf
import numpy as np
from .base_rbm import BaseRBM
from .utils import sample_bernoulli,sample_gaussian


class BernoulliRBM(BaseRBM):
    def __init__(self, *args, **kwargs):
        BaseRBM.__init__(self, *args, **kwargs)
    
    def _sample_v(self, v_means):
        return sample_bernoulli(v_means)

    def _get_free_energy(self, v):
        T1 = -tf.reduce_sum(tf.matmul(v, self._vb, transpose_b=True), axis=1)
        T2 = -tf.reduce_sum(tf.nn.softplus(tf.matmul(v, self._w) + self._hb), axis=1)
        fe = T1 + T2
        return fe

class GaussianRBM(BaseRBM):
    def __init__(self, sigma = 1., *args, **kwargs):
        BaseRBM.__init__(self, *args, **kwargs)
        self.sigma = sigma
        self._sigma_tmp = np.repeat(self.sigma, self.n_visible)
        self._sigma = tf.constant(self._sigma_tmp, dtype=self._dtype)
        self._sigma = tf.reshape(self._sigma, [1, self.n_hidden])

    def _sample_v(self, v_means):
        return sample_gaussian(v_means, self.sigma)
    
    def _get_free_energy(self, v):
        T1 = tf.divide(tf.reshape(self._vb, [1, self.n_visible]), self._sigma)
        T2 = tf.square(tf.subtract(v, T1))
        T3 = 0.5 * tf.reduce_sum(T2, axis=1)
        T4 = -tf.reduce_sum(tf.nn.softplus(self._propup(v) + self._hb), axis=1)
        fe = tf.reduce_mean(T3 + T4, axis=0)
        return fe