from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from tqdm import tqdm
from .plot_utils import im_plot
from .utils import np_sample_bernoulli, sample_bernoulli,make_list_from, assert_shape, assert_len, tf_count, batch_iter
from .rbm_ais import eval_logp


class BaseRBM:
    """
    References
    -----------
    [1] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT press, 2016.
    [2] Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." 
        Neural networks: Tricks of the trade. Springer, Berlin, Heidelberg, 2012. 599-619.
    [3] Fischer, Asja, and Christian Igel. "An introduction to restricted Boltzmann machines." 
        Iberoamerican Congress on Pattern Recognition. Springer, Berlin, Heidelberg, 2012.
    [4] Salakhutdinov, Ruslan, and Geoffrey E. Hinton. “Deep Boltzmann Machines.” 
        International Conference on Artificial Intelligence and Statistics, 2009, pp. 448–455.
    [5] Restricted Boltzmann Machines (RBMs), Deep Learning Tutorial
        url: http://deeplearning.net/tutorial/rbm.html
    """
    def __init__(self,
                 n_visible=784,
                 n_hidden=500,
                 W_init=None,
                 vb_init=None,
                 hb_init=None,
                 algorithm='CD',        #training algorithm:CD or PCD
                 precision='float32',
                 n_gibbs_step=1,
                 anneal_lr=False,
                 learning_rate=0.001,
                 momentum=0.9,
                 use_momentum=True,
                 max_epoch=150,
                 batch_size=10,
                 regularization='L2',
                 rl_coeff=1e-4,
                 sparsity=False,
                 sparsity_cost=None,
                 sparsity_target=None,
                 sparsity_damping=0.9,
                 n_feg=1,
                 n_validation=1,
                 dropout=None,
                 dropconnect=None,
                 dbm_first=False,
                 dbm_last=False,
                 sample_h_state=True,
                 sample_v_state=False,
                 save_after_each_epoch=True,
                 save_path='/data/experiments/',
                 n_batch_for_feg=50,
                 verbose=True,
                 img_shape=(28, 28)
                ):

        self._dtype = precision #single or double precision
        
        self.epoch_ = 0
        self.save_after_each_epoch = save_after_each_epoch
        self.save_path = save_path
        self.n_feg = n_feg
        self.n_validation = n_validation
        self.n_batch_for_feg = n_batch_for_feg
        self.verbose = verbose

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.img_shape = img_shape

        self.regularization = regularization
        self.rl_coeff = rl_coeff
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.r = None
        self.dbm_first = dbm_first
        self.dbm_last = dbm_last

        # self.learning_rate = make_list_from(learning_rate)
        self.learning_rate = make_list_from(learning_rate)
        self.momentum = make_list_from(momentum)
        self.use_momentum = use_momentum
        self.anneal_lr = anneal_lr
        self.n_gibbs_step = n_gibbs_step
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.epoch = 0

        self.mask = tf.ones([self.n_visible, self.n_hidden], dtype = self._dtype) # used for pruning

        self.algorithm = algorithm
        self._v_means = None         #reconstruction mean
        self._v_states = None       #reconstruction
        if self.algorithm == 'PCD':
            self.persistent = np.zeros((self.batch_size, self.n_hidden), dtype=self._dtype) #store the previous hidden state in PCD

        '''
        According to [2], the training goes less noisy and slightly faster, if
        sampling used for states of hidden units driven by the data, and probabilities
        for ones driven by reconstructions, and if probabilities (means) used for visible units,
        both driven by data and by reconstructions. It is therefore recommended to set
        these parameter to False (default).
        '''
        self.sample_h_state = sample_h_state
        self.sample_v_state = sample_v_state

        #weights init from existing weights
        self.W_init = W_init
        if self.W_init is not None:
            self.W_init = np.asarray(self.W_init)
            assert_shape(self, 'W_init', (self.n_visible, self.n_hidden))
        
        '''
        It is usually helpful to initialize the bias of visible unit i to log[p_i /(1 − p_i )]
        where p_i is the proportion of training vectors in which unit i is on. If this is not
        done, the early stage of learning will use the hidden units to make i turn on with
        a probability of approximately p_i.According to [2].
        '''
        self.vb_init = vb_init
        if self.vb_init is not None:
            self.vb_init = np.asarray(self.vb_init)
            self.vb_init = vb_init.reshape((1, self.n_visible))
            assert_shape(self, 'vb_init', (1, self.n_visible))

        self.hb_init = hb_init
        if self.hb_init is not None:
            self.hb_init = np.asarray(self.hb_init)
            self.hb_init = hb_init.reshape((1, self.n_hidden))
            assert_shape(self, 'hb_init', (1, self.n_hidden))

        #tf constant

        self._dbm_first = None
        self._dbm_last = None
        self._propup_multiplier = None
        self._propdown_multiplier = None

        self.sparsity = sparsity
        self._q_means = None
        self.sparsity_cost = sparsity_cost
        self.sparsity_target = sparsity_target
        self.sparsity_damping =  sparsity_damping

        #tf placeholder
        self._x = None
        self._learning_rate = None
        self._n_gibbs_step = None
        self._momentum = None

        #tf variables
        self._w = None
        self._vb = None
        self._hb = None

        self._dw = None
        self._dvb = None
        self._dhb = None

        # masks
        self._mask = None
        self._hidden_mask = None
        self._w_mask = None

        self._dw_cul = None #for pruning

        #tf train operation
        self._train_op = None       #train operation(CD-k)
        self._free_energy_op = None #free energy
        self._msre = None           #mean square reconstruction error
        self._pll = None            #pesudo likelihood
        self._merged = None         #tf summary merged
        self._train_writer = None   #tensorboard writer
        self._dW_cumulation = None  #for evaluation of parameters' derivatives

        self._make_tf_model()

        assert self._train_op is not None

        init = tf.global_variables_initializer()
        self._sess = tf.Session()
        self._sess.run(init)

    def _get_free_energy(self, v):  #depends on visible layer:Bernoulli or Gaussian
        pass
    
    def _sample_v(self, v_means):   #depends on visible layer:Bernoulli or Gaussian
        pass

    def _sample_h(self, h_means):
        return sample_bernoulli(h_means)

    def _h_means_given_v(self, v):
        if self.dropconnect is not None:
            return tf.nn.sigmoid(tf.matmul(v, tf.multiply(self._w, self._w_mask)) + self._hb)
        else:
            return tf.nn.sigmoid(tf.matmul(v, self._w) + self._hb)

    def _v_means_given_h(self, h):
        if self.dropconnect is not None:
            return tf.nn.sigmoid(tf.matmul(h, tf.multiply(self._w, self._w_mask), transpose_b=True) + self._vb)
        else:
            return tf.nn.sigmoid(tf.matmul(h, self._w, transpose_b=True) + self._vb)
    
    def _make_one_gibbs_step(self, h_states):
        v_states = v_means = self._v_means_given_h(h_states)
        if self.sample_v_state:
            v_states = self._sample_v(v_means)
        
        h1_states = h1_means = self._h_means_given_v(v_states)
        if self.sample_h_state:
            h1_states = self._sample_h(h1_means) if self.dropout is None else tf.multiply(self._sample_h(h1_means), self._hidden_mask)
        
        return v_states, v_means, h1_states, h1_means

    def _make_k_gibbs_steps(self, h_states):
        v_states = v_means = h_means = None
        for _ in range(self.n_gibbs_step):
            v_states, v_means, h_states, h_means = self._make_one_gibbs_step(h_states)
        return v_states, v_means, h_states, h_means
    
    def _make_k_gibbs_steps_variant(self, h_states, sample_interval):
        def cond(step, max_step, v_states, v_means, h_states, h_means):
            return step < max_step

        def body(step, max_step, v_states, v_means, h_states, h_means):
            v_states, v_means, h_states, h_means = self._make_one_gibbs_step(h_states)
            return step+1, max_step, v_states, v_means, h_states, h_means
        
        _, _, v_states, v_means, h_states, h_means = \
            tf.while_loop(cond=cond, body=body,
                          loop_vars=[tf.cast(tf.constant(0), dtype=self._dtype),
                                     tf.cast(sample_interval, dtype=self._dtype),
                                     tf.zeros_like(self._x, dtype=self._dtype),
                                     tf.zeros_like(self._x, dtype=self._dtype),
                                     h_states,
                                     tf.zeros_like(h_states)],
                          back_prop=False,
                          parallel_iterations=1)

        return v_states, v_means, h_states, h_means

    def _make_gibbs_chain(self, h_states):
        if self.anneal_lr:
            return self._make_k_gibbs_steps_variant(h_states, self._n_gibbs_step)
        else:
            return self._make_k_gibbs_steps(h_states)

    def sample(self, v_init, sample_interval, max_steps, save_path):
        if not os.path.isdir(self.save_path + save_path):
            os.mkdir(self.save_path + save_path)
        im_plot(v_init, shape=self.img_shape, n_rows=10, n_cols=10, title='samples generated from rbm after 0 Gibbs steps')
        plt.savefig(self.save_path + save_path + '/0.pdf')
        plt.close()
        v_states = self._sample_v(v_init) if self.sample_v_state else v_init
        h_states = self._sample_h(self._h_means_given_v(v_states))
        for i in range(max_steps // sample_interval):
            v_states, v_means, h_states, h_means = self._make_k_gibbs_steps_variant(h_states, sample_interval)
            im_plot(self._sess.run(v_means, feed_dict={self._x: np.zeros((100 ,self.n_visible)), self._hidden_mask:np.ones((100, self.n_hidden)), self._w_mask:np.ones([self.n_visible, self.n_hidden])}), \
            n_rows=10, n_cols=10, shape=self.img_shape, title='samples generated from rbm after %d Gibbs steps' % ((i+1)*sample_interval))
            plt.savefig(self.save_path + save_path + '/%d.pdf' % ((i+1)*sample_interval))
            plt.close()

    def sample_sequential(self, v_init, sample_interval, max_steps, save_path):
        img_sequential = v_init
        if not os.path.isdir(self.save_path + save_path):
            os.mkdir(self.save_path + save_path)
        v_states = self._sample_v(v_init) if self.sample_v_state else v_init
        h_states = self._sample_h(self._h_means_given_v(v_states))
        for i in range(max_steps // sample_interval):
            v_states, v_means, h_states, h_means = self._make_k_gibbs_steps_variant(h_states, sample_interval)
            v_next = self._sess.run(v_means, feed_dict={self._x: np.zeros((100 ,self.n_visible)), self._hidden_mask:np.ones((100, self.n_hidden)), self._w_mask:np.ones([self.n_visible, self.n_hidden])})
            img_sequential = np.vstack((img_sequential, v_next))
        im_plot(img_sequential, n_rows=10, n_cols=10, shape=self.img_shape, title='samples generated from rbm')
        plt.savefig(self.save_path + save_path + '/sequential_img.pdf')
        plt.close()

    def display_filters(self, num, n_rows, n_cols, save_name=None): 
        if not os.path.isdir(self.save_path + 'filters'):
            os.mkdir(self.save_path + 'filters')
        if num > self.n_hidden:
            raise ValueError('number of filters to display cannot be larger than number of hidden units')
        else:
            w = self._w.eval(session=self._sess)
            im_plot(np.transpose(w)[:num,], n_rows=n_rows, n_cols=n_cols, shape=self.img_shape)
            if save_name is not None:
                plt.savefig(self.save_path + 'filters/' + save_name + '.pdf')
            else:
                plt.savefig(self.save_path + 'filters/filters_at_epoch_%d.pdf' % self.epoch_)
            plt.close()

    def _make_constants(self):
        if self.sparsity:
            if self.hb_init is None:
                self.hb_init = np.repeat(np.log(self.sparsity_target / (1. - self.sparsity_target)), self.n_hidden)
                self.hb_init = np.reshape(self.hb_init, [1, self.n_hidden])

        self._dbm_first = tf.constant(self.dbm_first, dtype=tf.bool, name='is_dbm_first')
        self._dbm_last = tf.constant(self.dbm_last, dtype=tf.bool, name='is_dbm_last')
        t = tf.constant(1., dtype=self._dtype, name="1")
        t1 = tf.cast(self._dbm_first, dtype=self._dtype)
        self._propup_multiplier = tf.identity(tf.add(t1, t), name='propup_multiplier')
        t2 = tf.cast(self._dbm_last, dtype=self._dtype)
        self._propdown_multiplier = tf.identity(tf.add(t2, t), name='propdown_multiplier')

    def _make_placeholders(self):
        with tf.name_scope('input_data'):
            self._x = tf.placeholder(self._dtype, [None, self.n_visible], name='x_batch')
            self._learning_rate = tf.placeholder(self._dtype, [], name='learning_rate')
            self._momentum = tf.placeholder(self._dtype, [], name='momentum')
            self._n_gibbs_step = tf.placeholder(self._dtype, [], name='n_gibbs_step')
            self._hidden_mask = tf.placeholder(self._dtype, [None, self.n_hidden], name='dropout')
            self._w_mask = tf.placeholder(self._dtype, [self.n_visible, self.n_hidden], name='dropconnect')
    
    def _make_variables(self):
        if self.W_init is not None:
            W_init = tf.constant(self.W_init, dtype=self._dtype)
        else:
            W_init = tf.random_normal([self.n_visible, self.n_hidden], mean=0.0, stddev=0.01, dtype=self._dtype)

        if self.vb_init is not None:
            vb_init = tf.constant(self.vb_init, dtype=self._dtype)
        else:
            vb_init = tf.zeros([1, self.n_visible], dtype=self._dtype)

        if self.hb_init is not None:
            hb_init = tf.constant(self.hb_init, dtype=self._dtype)
        else:
            hb_init = tf.zeros([1, self.n_hidden], dtype=self._dtype)

        with tf.name_scope('weights'):
            self._w = tf.Variable(W_init, dtype=self._dtype, name='w')
            self._vb = tf.Variable(vb_init, dtype=self._dtype, name='vb')
            self._hb = tf.Variable(hb_init, dtype=self._dtype, name='hb')

        # tf.summary.histogram('w', self._w)
        # tf.summary.histogram('vb', self._vb)
        # tf.summary.histogram('hb', self._hb)

        dW_init = tf.zeros([self.n_visible, self.n_hidden], dtype=self._dtype)
        dvb_init = tf.zeros([1, self.n_visible], dtype=self._dtype)
        dhb_init = tf.zeros([1, self.n_hidden], dtype=self._dtype)

        with tf.name_scope('derivatives'):
            self._dw = tf.Variable(dW_init, dtype=self._dtype, name='dw')
            self._dvb = tf.Variable(dvb_init, dtype=self._dtype, name='dvb')
            self._dhb = tf.Variable(dhb_init, dtype=self._dtype, name='dhb')

        self._mask = tf.Variable(self.mask, dtype=self._dtype, name='mask')

        # self._dw_cul = tf.Variable(tf.zeros((self.n_visible, self.n_hidden), dtype=self._dtype))

        # tf.summary.histogram('dw', self._dw)
        # tf.summary.histogram('dvb', self._dvb)
        # tf.summary.histogram('dhb', self._dhb)        

        if self.sparsity:
            self._q_means = tf.Variable(tf.zeros([self.n_hidden], dtype=self._dtype), dtype=self._dtype, name='q_means')

    def _make_train_op(self):
        
        x0 = self._x if not self.sample_v_state else sample_bernoulli(self._x)
        h0_means = self._h_means_given_v(x0)
        h0_states = self._sample_h(h0_means)

        if self.dropout is not None:
            h0_states = tf.multiply(h0_states, self._hidden_mask)

        if self.algorithm == 'CD':
            self._v_states, self._v_means, _, h_means = self._make_gibbs_chain(h0_states)
            dW_negative = tf.matmul(self._v_states, h_means, transpose_a=True)
            dvb = tf.reduce_mean(x0 - self._v_states, axis=0)

        elif self.algorithm == 'PCD':
            v_samples, _, self.persistent, h_means = self._make_gibbs_chain(self.persistent)
            _, self._v_means, _, _ = self._make_gibbs_chain(h0_states)

            dW_negative = tf.matmul(v_samples, h_means, transpose_a=True)
            dvb = tf.reduce_mean(x0 - v_samples, axis=0)
            
        dW_positive = tf.matmul(x0, h0_means, transpose_a=True)
        dW = (dW_positive - dW_negative) / tf.cast(tf.shape(x0)[0], self._dtype)

        if self.regularization == 'L2':
            dW = dW - 2 * self.rl_coeff * self._w
        elif self.regularization == 'L1':
            dW = dW - self.rl_coeff * tf.sign(self._w)
        dhb = tf.reduce_mean(h0_means - h_means, axis=0)

        if self.sparsity:
            q_means = tf.reduce_sum(h_means, axis=0)
            q_update = self._q_means.assign(self.sparsity_damping * self._q_means + \
                                            (1 - self.sparsity_damping)*q_means)
            sparsity_penalty = self.sparsity_cost * (q_update - self.sparsity_target)
            dW -= sparsity_penalty
            dhb -= sparsity_penalty
        
        if self.use_momentum:
            dW_update = self._dw.assign(self._momentum * self._dw + self._learning_rate * dW)
            dvb_update = self._dvb.assign(self._momentum * self._dvb + self._learning_rate * dvb)
            dhb_update = self._dhb.assign(self._momentum * self._dhb + self._learning_rate * dhb)
        else:
            dW_update = self._dw.assign(self._learning_rate * dW)
            dvb_update = self._dvb.assign(self._learning_rate * dvb)
            dhb_update = self._dhb.assign(self._learning_rate * dhb)
        
        # self._dW_cumulation = self._dw_cul.assign_add(dW)
        dW_update = tf.multiply(dW_update, self._mask)
        if self.sparsity:
            dW_update = tf.multiply(dW_update, self._w_mask)        
        W_update = self._w.assign_add(dW_update)
        vb_update = self._vb.assign_add(dvb_update)
        hb_update = self._hb.assign_add(dhb_update)

        with tf.name_scope('training_step'):
            self._train_op = tf.group(W_update, vb_update, hb_update)

        with tf.name_scope('mean_square_reconstruction_error'):        
            self._msre = tf.reduce_mean(tf.square(self._x - self._v_means))
            self._cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.multiply(self._x, tf.log(self._v_means)) + tf.multiply(1 - self._x, tf.log(1 - self._v_means)), axis=1))
 
        '''
        Since reconstruction error is fairly poor measure of performance,
        as this is not what CD-k learning algorithm aims to minimize [2],
        compute (per sample average) pseudo-loglikelihood (proxy to likelihood)
        instead, which not only is much more cheaper to compute, but also
        learning with PLL is asymptotically consistent [1].
        More specifically, PLL computed using approximation as in [5].
        '''
        with tf.name_scope('pseudo_loglik'):
            x = tf.round(self._x)
            # x = self._x
            x_ = tf.identity(x)
            batch_size = tf.shape(x)[0]
            pll_rand = tf.random_uniform([batch_size], minval=0, maxval=self.n_visible, dtype=tf.int32)
            ind = tf.transpose([tf.range(batch_size), pll_rand])
            m = tf.SparseTensor(indices=tf.to_int64(ind), values=tf.ones_like(pll_rand, dtype=self._dtype), dense_shape=tf.to_int64(tf.shape(x_)))
            x_ = tf.multiply(x_, -tf.sparse_tensor_to_dense(m, default_value=-1))
            x_ = tf.sparse_add(x_, m)
            x_ = tf.identity(x_, name='x_corrupted')

            self._pll = tf.reduce_mean(tf.cast(self.n_visible, dtype=self._dtype) * (-tf.nn.softplus(self._get_free_energy(x) - self._get_free_energy(x_))))

        with tf.name_scope('free_energy_operation'):
            self._free_energy_op = tf.reduce_sum(self._get_free_energy(self._x))

    def _run_feg(self, data_x, data_x_val):
        """Calculate difference between average free energies of subsets
        of validation and training sets to monitor overfitting,
        as proposed in [2]. If the model is not overfitting at all, this
        quantity should be close to zero. Once this value starts
        growing, the model is overfitting and the value ("free energy gap")
        represents the amount of overfitting.
        """
        batch_size = self.batch_size
        train_fes = []
        n_data = data_x.shape[0]
        n_batch = n_data // self.batch_size + (n_data % self.batch_size > 0)
        for i in range(self.n_batch_for_feg):
            if i == n_batch - 1:
                batch_x = data_x[i*self.batch_size::]
            else:
                batch_x = data_x[i*self.batch_size:i*batch_size+batch_size]
            train_fe = self._sess.run(self._free_energy_op,
                                            feed_dict={self._x: batch_x})
            train_fes.append(train_fe)

        val_fes = []
        n_data = data_x_val.shape[0]
        n_batch = n_data // self.batch_size + (n_data % self.batch_size > 0)
        for i in range(self.n_batch_for_feg):
            if i == n_batch - 1:
                batch_x = data_x_val[i*batch_size::]
            else:
                batch_x = data_x_val[i*batch_size:i*batch_size+batch_size]
            val_fe = self._sess.run(self._free_energy_op,
                                            feed_dict={self._x: batch_x})
            val_fes.append(val_fe)

        feg = np.mean(val_fes) - np.mean(train_fes)
        return feg
    
    def _run_validation(self, data_x_val):

        val_pll = []
        val_msre = []
        val_cross_entropy =[]
        # batch_size = self.batch_size
        # n_data_val = data_x_val.shape[0]
        # n_batch_val = n_data_val // self.batch_size + (n_data_val % self.batch_size > 0)    
        # # for i in range(n_batch_val):
        # #     if i == n_batch_val - 1:
        # #         batch_x_val = data_x_val[i*batch_size::]
        # #     else:
        # #         batch_x_val = data_x_val[i*batch_size:i*batch_size+batch_size]
        pll, msre, cross_entropy = self._sess.run([self._pll, self._msre, self._cross_entropy], feed_dict={self._x: data_x_val})
            
        val_pll.append(pll)
        val_msre.append(msre)
        val_cross_entropy.append(cross_entropy)
        
        return np.mean(val_pll), np.mean(val_msre), np.mean(val_cross_entropy)

    def _make_tf_model(self):
        self._make_constants()
        self._make_placeholders()
        self._make_variables()
        self._make_train_op()

    def _partial_fit(self, batch_x, lr=None):
        d = {}
        if self.anneal_lr:
            d['n_gibbs_step'] = ceil((self.epoch+1) / 10) if ceil((self.epoch+1) / 10) <=25 else 25
            d['learning_rate'] = self.learning_rate[0] / (d['n_gibbs_step'])
        else:
            d['n_gibbs_step'] = self.n_gibbs_step
        
            if lr is None:
                d['learning_rate'] = self.learning_rate[min(self.epoch_, len(self.learning_rate)-1)]
            else:
                d['learning_rate'] = lr
        
        if self.dropout is not None:
            d['dropout'] = np_sample_bernoulli(np.repeat(np.repeat(self.dropout, self.n_hidden).reshape((1, self.n_hidden)), batch_x.shape[0], axis=0))
        
        if self.dropconnect is not None:
            d['dropconnect'] = np_sample_bernoulli(np.repeat(np.repeat(self.dropconnect, self.n_hidden).reshape((1, self.n_hidden)), self.n_visible, axis=0)).astype(self._dtype)

        d['momentum'] = self.momentum[min(self.epoch, len(self.momentum)-1)]
        d['x_batch'] = batch_x

        feed_dict = {}
        for k,v in d.items():
            feed_dict['input_data/{0}:0'.format(k)] = v

        return self._sess.run(self._train_op, feed_dict=feed_dict)

    def fit(self,
            data_x,
            data_x_test,
            retrain=False,
            lr=None,
            epoch=None
            ):
         
        epochs = epoch if epoch is not None else self.max_epoch

        for self.epoch_ in range(epochs):
            for batch_x in batch_iter(data_x, self.batch_size, self.verbose, self.epoch_): 

                if self.sample_v_state:
                    batch_x = np_sample_bernoulli(batch_x)
                    
                if lr is None:
                    self._partial_fit(batch_x) 
                else:
                    self._partial_fit(batch_x, lr)

            if not retrain:
                if self.n_hidden >=100:    
                    self.display_filters(100, 10, 10, 'epoch-%d' % self.epoch_)
                else:
                    self.display_filters(20, 2, 10, 'epoch-%d' % self.epoch_)
            
            if self.save_after_each_epoch:
                 self._save_weights('my-model-{}'.format(self.epoch_))
            
            self.epoch = self.epoch + 1
            
            # used to watch the training progress at every 10 epochs
            # if self.epoch % 10 == 0:
            #     if self.n_hidden < 30:
            #         logZ, avg_logp = eval_logp(self._sess, True, self._w, self._vb, self._hb, data_x, data_x_test, self._dtype, 100, 100)
            #     else:
            #         logZ, avg_logp = eval_logp(self._sess, False, self._w, self._vb, self._hb, data_x, data_x_test, self._dtype, 100, 100)
                
            #     print('epoch%d:test logZ:%f,avg logp:%f' % (self.epoch, logZ, avg_logp))

            #     if self.n_hidden < 30:
            #         logZ, avg_logp = eval_logp(self._sess, True, self._w, self._vb, self._hb, data_x, data_x, self._dtype, 100, 100)
            #     else:
            #         logZ, avg_logp = eval_logp(self._sess, False, self._w, self._vb, self._hb, data_x, data_x, self._dtype, 100, 100)
                
            #     print('epoch%d:training logZ:%f,avg logp:%f' % (self.epoch, logZ, avg_logp))


        if self.n_hidden < 30:
            logZ, avg_logp = eval_logp(self._sess, True, self._w, self._vb, self._hb, data_x, data_x_test, self._dtype, 100, 100)
        else:
            logZ, avg_logp = eval_logp(self._sess, False, self._w, self._vb, self._hb, data_x, data_x_test, self._dtype, 100, 100)

        return logZ, avg_logp

    def pruning_weight(self, data_x, data_x_test, sparsity_ratio, threshold=None):

        if threshold is None:
            sorted_w = tf.contrib.framework.sort(tf.reshape(tf.abs(self._w), [-1]))
            threshold_w = sorted_w[tf.cast(tf.cast(tf.size(sorted_w), dtype=self._dtype) * sparsity_ratio, tf.int32)]
            print('threshold_w:', self._sess.run(threshold_w))
        else:
            threshold_w = threshold
            print('threshold_w:',threshold_w)

        cond_w = tf.less(tf.abs(self._w), threshold_w)
        self.mask = tf.where(cond_w, tf.zeros_like(self._w, dtype=self._dtype), self._mask)
        mask_update = self._mask.assign(self.mask)
        pruning_op = self._w.assign(tf.where(cond_w, tf.zeros_like(self._w, dtype=self._dtype), self._w))
        self._sess.run([mask_update, pruning_op])

        for col in range(self.n_hidden):
            if self._sess.run(tf.reduce_sum(self._w[:,col])) == 0.:
                self._sess.run(self._hb[0, col].assign(tf.cast(0., dtype=self._dtype)))

        if self.n_hidden < 30:

            logZ, avg_logp = eval_logp(self._sess, True, self._w, self._vb, self._hb, data_x, data_x_test, self._dtype, 100, 100)
        else:
            logZ, avg_logp = eval_logp(self._sess, False, self._w, self._vb, self._hb, data_x, data_x_test, self._dtype, 100, 100)

        sparsity = self._sess.run(tf.cast(tf_count(self._mask, tf.cast(0, dtype=self._dtype)), dtype=self._dtype) / tf.cast(self.n_visible*self.n_hidden, self._dtype))
        return logZ, avg_logp, sparsity

    def cal_sparsity(self):
        return self._sess.run(tf.cast(tf_count(self._mask, tf.cast(0, dtype=self._dtype)), dtype=self._dtype) / tf.cast(self.n_visible*self.n_hidden, self._dtype))

    def plot_w_hist(self, w, bins, name, xlabel, ylabel, title):
        plt.hist(self._sess.run(w), bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.savefig(self.save_path + name + '.pdf')
        # plt.close()

    def reinit_weight(self):
        W_init = tf.random_normal([self.n_visible, self.n_hidden], mean=0.0, stddev=0.01, dtype=self._dtype)
        vb_init = tf.zeros([1, self.n_visible], dtype=self._dtype)
        hb_init = tf.zeros([1, self.n_hidden], dtype=self._dtype)
        w_reinit = self._w.assign(tf.multiply(W_init, self._mask))
        vb_reinit = self._vb.assign(vb_init)
        hb_reinit = self._hb.assign(hb_init)
        self._sess.run([w_reinit, vb_reinit, hb_reinit])
        
    def reset_mask(self):
        self._sess.run(self._mask.assign(tf.ones([self.n_visible, self.n_hidden], dtype=self._dtype)))

    def recover_mask(self):
        cond_zero = tf.equal(self._w, tf.cast(0, dtype=self._dtype))
        self._sess.run(self._mask.assign(tf.where(cond_zero, tf.zeros_like(self._mask), self._mask)))

    def _get_weights(self):
        return self._sess.run(self._w),\
            self._sess.run(self._vb),\
            self._sess.run(self._hb)

    def _save_weights(self, filename):
        saver = tf.train.Saver({'w': self._w,
                                'vb': self._vb,
                                'hb': self._hb})
        return saver.save(self._sess, self.save_path + filename)

    def _set_weights(self, w, vb, hb):
        self._sess.run(self._w.assign(w))
        self._sess.run(self._vb.assign(vb))
        self._sess.run(self._hb.assign(hb))

    def _load_weights(self, filename):
        saver = tf.train.Saver({'w': self._w,
                                'vb': self._vb,
                                'hb': self._hb})
        saver.restore(self._sess, self.save_path + filename)

    def _get_weights_mask(self):
        return self._sess.run(self._w),\
            self._sess.run(self._vb),\
            self._sess.run(self._hb),\
            self._sess.run(self._mask)

    def _save_weights_mask(self, filename):
        saver = tf.train.Saver({'w': self._w,
        
                                'vb': self._vb,
                                'hb': self._hb,
                                'mask': self._mask})
        return saver.save(self._sess, self.save_path + filename)

    def _set_weights_mask(self, w, vb, hb, mask):
        self._sess.run(self._w.assign(w))
        self._sess.run(self._vb.assign(vb))
        self._sess.run(self._hb.assign(hb))
        self._sess.run(self._mask.assign(mask))

    def _load_weights_mask(self, filename):
        saver = tf.train.Saver({'w': self._w,
                                'vb': self._vb,
                                'hb': self._hb,
                                'mask':self._mask})
        saver.restore(self._sess, self.save_path + filename)
