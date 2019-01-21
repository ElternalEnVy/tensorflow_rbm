import numpy as np
import tensorflow as tf
from .utils import np_sample_bernoulli,sample_bernoulli

'''
reference:
----------
[1] Salakhutdinov, Ruslan, and Iain Murray. "On the quantitative analysis of deep belief networks." 
    Proceedings of the 25th international conference on Machine learning. ACM, 2008.
        
This code is implemented according to [1],using Annealed Importance Sampling to approximate
the partition function.
'''

def base_rate(sess, data_x, type, batch_size):
    [n_data, n_visible] = data_x.shape
    batch_size = batch_size
    n_batch = n_data // batch_size + (n_data % batch_size > 0)

    count_int_init = np.zeros([1, n_visible], dtype=type)
    count_int = tf.Variable(count_int_init, dtype=type)
    x = tf.placeholder(dtype=type, shape=[None, n_visible])
    sess.run(tf.variables_initializer([count_int]))
    update = count_int.assign_add(tf.reshape(tf.reduce_sum(x, axis=0), [1, n_visible]))

    for i in range(n_batch):
        if i == n_batch - 1:
            batch_data = data_x[i*batch_size::]
        else:
            batch_data = data_x[i*batch_size:i*batch_size+batch_size]
        sess.run(update, feed_dict={x: batch_data})
    lp = 5
    p_int = (count_int + lp*n_batch) / (batch_size*n_batch+lp*n_batch)
    log_base_rate = tf.log(p_int) - tf.log(1-p_int)
    res = sess.run(log_base_rate)
    return res

'''
This is the setting(14500 intermediate distributions) that [1] used to approximate 
the partition function on MNIST.We use this function in our experiments.
'''
def AIS(sess, w, vb, hb, type, log_base_rate, num_run):
    [n_visible, n_hidden] = sess.run(tf.shape(w))
    beta0 = np.linspace(0, 0.5, 500, endpoint=False)
    beta1 = np.linspace(0.5, 0.9, 4000, endpoint=False)
    beta2 = np.linspace(0.9, 1.0, 10001, endpoint=False)
    beta = np.append(np.append(beta0, beta1), beta2) 

    vb_base = np.repeat(log_base_rate, num_run, axis=0)
    negdata_init = np_sample_bernoulli(1./(1+np.exp(-vb_base)))
    negdata = tf.Variable(negdata_init, dtype=type)
    log_base_rate = tf.constant(log_base_rate, dtype=type)

    logww_init = np.zeros([num_run, 1], dtype=type)
    logww = tf.Variable(logww_init, dtype=type)

    sess.run(tf.variables_initializer([negdata, logww]))

    Wh = tf.matmul(negdata, w) + hb
    Bv_base = tf.matmul(negdata, log_base_rate, transpose_b=True)
    Bv = tf.matmul(negdata, vb, transpose_b=True)

    bb = tf.placeholder(type, shape=(), name='beta')
    expWh = tf.exp(bb*Wh)
            
    delta_postive = (1-bb)*Bv_base + bb*Bv + tf.reshape(tf.reduce_sum(tf.log(1+expWh), axis=1), [num_run, 1])
    update_postive = logww.assign_add(delta_postive)
    delta_negtive = -(1-bb)*Bv_base - bb*Bv - tf.reshape(tf.reduce_sum(tf.log(1+expWh), axis=1), [num_run, 1])
    update_negtive = logww.assign_add(delta_negtive)
            
    poshidprobs = expWh/(1 + expWh)
    poshidstates = sample_bernoulli(poshidprobs)

    T = tf.nn.sigmoid((1-bb)*vb_base + bb*(tf.matmul(poshidstates, w, transpose_b=True)+vb))
    T_sample = negdata.assign(sample_bernoulli(T))

    
    sess.run(update_negtive, feed_dict={bb: np.array(0)})  
    
    for i in range(1, len(beta)):
        sess.run(update_postive, feed_dict={bb: beta[i]})
        sess.run(T_sample, feed_dict={bb: beta[i]})
        sess.run(update_negtive, feed_dict={bb: beta[i]})

    sess.run(update_postive, feed_dict={bb: np.array(1)})

    logZZ_base = tf.reduce_sum(tf.nn.softplus(log_base_rate)) + n_hidden*tf.cast(tf.log(2.), dtype=type)
    alpha_s = tf.reduce_max(logww) - tf.log(np.finfo(type).max) / 2      
    r_AIS = tf.log(tf.reduce_sum(tf.exp(logww-alpha_s))) + alpha_s - tf.log(tf.constant(num_run, dtype=type))
    logZZ_est = r_AIS + logZZ_base
    res = sess.run(logZZ_est)
    return res

def AIS_custom(sess, w, vb, hb, type, log_base_rate, num_run, num_intermediate):
    [n_visible, n_hidden] = sess.run(tf.shape(w))
    beta = np.linspace(0, 1.0, num_intermediate, endpoint=False)

    vb_base = np.repeat(log_base_rate, num_run, axis=0)
    negdata_init = np_sample_bernoulli(1./(1+np.exp(-vb_base)))
    negdata = tf.Variable(negdata_init, dtype=type)
    log_base_rate = tf.constant(log_base_rate, dtype=type)

    logww_init = np.zeros([num_run, 1], dtype=type)
    logww = tf.Variable(logww_init, dtype=type)

    sess.run(tf.variables_initializer([negdata, logww]))

    Wh = tf.matmul(negdata, w) + hb
    Bv_base = tf.matmul(negdata, log_base_rate, transpose_b=True)
    Bv = tf.matmul(negdata, vb, transpose_b=True)

    bb = tf.placeholder(type, shape=(), name='beta')
    expWh = tf.exp(bb*Wh)
            
    delta_postive = (1-bb)*Bv_base + bb*Bv + tf.reshape(tf.reduce_sum(tf.log(1+expWh), axis=1), [num_run, 1])
    update_postive = logww.assign_add(delta_postive)
    delta_negtive = -(1-bb)*Bv_base - bb*Bv - tf.reshape(tf.reduce_sum(tf.log(1+expWh), axis=1), [num_run, 1])
    update_negtive = logww.assign_add(delta_negtive)
            
    poshidprobs = expWh/(1 + expWh)
    poshidstates = sample_bernoulli(poshidprobs)

    T = tf.nn.sigmoid((1-bb)*vb_base + bb*(tf.matmul(poshidstates, w, transpose_b=True)+vb))
    T_sample = negdata.assign(sample_bernoulli(T))

    
    sess.run(update_negtive, feed_dict={bb: np.array(0)})  
    
    for i in range(1, len(beta)):
        sess.run(update_postive, feed_dict={bb: beta[i]})
        sess.run(T_sample, feed_dict={bb: beta[i]})
        sess.run(update_negtive, feed_dict={bb: beta[i]})

    sess.run(update_postive, feed_dict={bb: np.array(1)})

    logZZ_base = tf.reduce_sum(tf.nn.softplus(log_base_rate)) + n_hidden*tf.cast(tf.log(2.), dtype=type)
    alpha_s = tf.reduce_max(logww) - tf.log(np.finfo(type).max) / 2      
    r_AIS = tf.log(tf.reduce_sum(tf.exp(logww-alpha_s))) + alpha_s - tf.log(tf.constant(num_run, dtype=type))
    logZZ_est = r_AIS + logZZ_base
    res = sess.run(logZZ_est)
    return res

def avg_log_p(sess, w, vb, hb, type, logZ, data_test, batch_size):
    [n_visible, n_hidden] = sess.run(tf.shape(w))
    n_data = data_test.shape[0]
    n_batch = n_data // batch_size + (n_data % batch_size > 0)
    p_un = tf.Variable(tf.zeros((), dtype=type))
    sess.run(tf.variables_initializer([p_un]))
    x = tf.placeholder(dtype=type, shape=[None, n_visible])

    delta_1 = tf.reduce_sum(tf.matmul(x, vb, transpose_b=True)) 
    delta_2 = tf.reduce_sum(tf.nn.softplus(tf.matmul(x, w) + hb))
    delta = delta_1 + delta_2
    update = p_un.assign_add(delta)

    for i in range(n_batch):
        if i == n_batch - 1:
            batch_data = data_test[i*batch_size::]
        else:
            batch_data = data_test[i*batch_size:i*batch_size+batch_size]
    
        sess.run(update, feed_dict={x: batch_data})
    avg_log_p = tf.reduce_sum(p_un) / n_data - logZ
    
    res = sess.run(avg_log_p)
    return res

def cal_true_logZ(sess, w, vb, hb, type):
    
    Z = tf.Variable(tf.zeros((), dtype=type))
    sess.run(tf.variables_initializer([Z]))
    [n_visible, n_hidden] = sess.run(tf.shape(w))
    h_states_array = np.zeros([2**20, n_hidden], dtype=type)
    h_state = tf.placeholder(type, [None, n_hidden])
    logZ_state = tf.placeholder(type, [None, 1])
    op = tf.reduce_sum(tf.matmul(h_state, hb, transpose_b=True), axis=1) + tf.reduce_sum(tf.nn.softplus(tf.matmul(h_state, w, transpose_b=True) + vb), axis=1)
    
    max_val = 0
    # for i in range(2**n_hidden):
    #     index = np.array(i)
    #     state = (((index & (1 << np.arange(n_hidden)))) > 0).astype(type)
    #     h_states_array[i] = state
    for i in range(2**(n_hidden-20)):
        for j in range(2**20):
            index = np.array(i*2**20+j)
            state = (((index & (1 << np.arange(n_hidden)))) > 0).astype(type)
            h_states_array[j] = state
        logZ_array = sess.run(op, feed_dict={h_state: h_states_array}).reshape([2**20, 1])
        temp_max = sess.run(tf.reduce_max(logZ_array))
        if i == 0:
            max_val = temp_max
        else:
            if temp_max > max_val:
                max_val = temp_max

    alpha_s = max_val - tf.log(np.finfo(type).max) / 2
    Z_add = Z.assign_add(tf.reduce_sum(tf.exp(logZ_state-alpha_s)))

    for i in range(2**(n_hidden-20)):
        for j in range(2**20):
            index = np.array(i*2**20+j)
            state = (((index & (1 << np.arange(n_hidden)))) > 0).astype(type)
            h_states_array[j] = state
        logZ_array = sess.run(op, feed_dict={h_state: h_states_array}).reshape([2**20, 1])
        sess.run(Z_add, feed_dict={logZ_state: logZ_array})
    res = sess.run(tf.log(Z) + alpha_s)
    return res

def eval_logp(sess, real, w, vb, hb, data_x, data_x_test, type, batch_size, num_run):
    log_base_rate = base_rate(sess, data_x, type, batch_size)
    if real:
        logZ = cal_true_logZ(sess, w, vb, hb, type)
    else:
        logZ = AIS(sess, w, vb, hb, type, log_base_rate, num_run)
    avg_logp = avg_log_p(sess, w, vb, hb, type, logZ, data_x_test, batch_size)
    return logZ, avg_logp

def eval_logp_custom(sess, real, w, vb, hb, data_x, data_x_test, type, batch_size, num_run, num_intermediate):
    log_base_rate = base_rate(sess, data_x, type, batch_size)
    if real:
        logZ = cal_true_logZ(sess, w, vb, hb, type)
    else:
        logZ = AIS_custom(sess, w, vb, hb, type, log_base_rate, num_run, num_intermediate)
    avg_logp = avg_log_p(sess, w, vb, hb, type, logZ, data_x_test, batch_size)
    return logZ, avg_logp

def eval_logp_dbn(sess, logZ, entropy, approximation):
    return sess.run(tf.reduce_mean(approximation + entropy)-logZ)