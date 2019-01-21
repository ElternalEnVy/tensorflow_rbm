import numpy as np
import tensorflow as tf
from tfrbm.rbm import BernoulliRBM
from tfrbm.dataset import load_OCR_letters
from tfrbm.utils import np_sample_bernoulli

X, labels = load_OCR_letters(mode='train')
X_test, labels_test = load_OCR_letters(mode='test')

X = X.astype('float32') 
X_test = X_test.astype('float32')

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

labels = indices_to_one_hot(labels, 26)
labels_test = indices_to_one_hot(labels_test, 26)

def pruning_iter_probability(rbm, iterations, prob, X, X_test):
    print('\n\niterative pruning and retraining in a probability way,without re-initialization\n\n')
    sparsity = 0.
    for i in range(iterations):
        print('\niteration-%d' % i)
        logZ, avg_logp, sparsity = rbm.pruning_weight(X, X_test, sparsity + (1. - sparsity)* prob)
        print('after pruning,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))
        logZ, avg_logp = rbm.fit(X, X_test, retrain=True)
        print('after pruning and retraining,sparsity:%f,logZ:%f,average_logp:%f' % (sparsity, logZ, avg_logp))    


def unsupervised_pretrain():
    weights = []
    masks = []

    g_1 = tf.Graph()
    with g_1.as_default():
        rbm1 = BernoulliRBM(n_visible=128, n_hidden=1000, precision='float32', algorithm='CD', anneal_lr=False, learning_rate=0.01, \
                    use_momentum=True, momentum=[0.5,0.5,0.5,0.5,0.5,0.9], max_epoch=50, batch_size=100, regularization='L2', \
                    rl_coeff=1e-4, sample_h_state=True, sample_v_state=True, save_path='/data/experiments/pruning_rbm/ocr_letters/classification/', save_after_each_epoch=False, verbose=False)
        
        print('RBM1:')    

        logZ, avg_logp = rbm1.fit(X, X_test, retrain=True)
        print('baseline:logZ:%f, average logp:%f' % (logZ, avg_logp))
        pruning_iter_probability(rbm1, 4, 0.3, X, X_test)
        
        weights.append(rbm1._sess.run(rbm1._w))
        weights.append(rbm1._sess.run(rbm1._hb))
        masks.append(rbm1._sess.run(rbm1._mask))

        X_1 = rbm1._sess.run(rbm1._h_means_given_v(X))
        X_1_binary = rbm1._sess.run(rbm1._sample_h(rbm1._h_means_given_v(X)))
        X_1_test_binary = rbm1._sess.run(rbm1._sample_h(rbm1._h_means_given_v(X_test)))

        rbm1._sess.close()

    g_2 = tf.Graph()
    with g_2.as_default():
        rbm2 = BernoulliRBM(n_visible=1000, n_hidden=1000, precision='float32', algorithm='CD', anneal_lr=False, learning_rate=0.01, \
                    use_momentum=True, momentum=[0.5,0.5,0.5,0.5,0.5,0.9], max_epoch=50, batch_size=100, regularization='L2', \
                    rl_coeff=1e-4, sample_h_state=True, sample_v_state=True, save_path='/data/experiments/pruning_rbm/ocr_letters/classification/', save_after_each_epoch=False, verbose=False)

        print('RBM2:')

        logZ, avg_logp = rbm2.fit(X_1, X_1_test_binary, retrain=True)
        print('baseline:logZ:%f, average logp:%f' % (logZ, avg_logp))
        pruning_iter_probability(rbm2, 4, 0.3, X_1, X_1_test_binary)

        weights.append(rbm2._sess.run(rbm2._w))
        weights.append(rbm2._sess.run(rbm2._hb))
        masks.append(rbm2._sess.run(rbm2._mask))

        rbm2._sess.close()

    return weights, masks

def supervised_learn(weights, masks):
    # rbm layerwise pretrain weights
    w1 = tf.Variable(weights[0], dtype=tf.float32)
    b1 = tf.Variable(weights[1], dtype=tf.float32)

    w2 = tf.Variable(weights[2], dtype=tf.float32)
    b2 = tf.Variable(weights[3], dtype=tf.float32)

    # classifier weights
    w3 = tf.Variable(tf.random_normal([1000, 26], mean=0.0, stddev=0.01, dtype=tf.float32))
    b3 = tf.Variable(tf.zeros([1, 26], dtype=tf.float32))

    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_increment = global_step.assign_add(1)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                    50*42154, 0.1, staircase=True)
    # input data
    x = tf.placeholder(tf.float32, [None, 128], name='input')
    y = tf.placeholder(tf.int32, [None, 26], name='train_labels')

    # network architecture
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1 ,w2), b2))

    # output logits,loss and optimizer
    logits = tf.add(tf.matmul(layer_2, w3), b3)
    y_pred = tf.nn.softmax(logits)
    y_one_hot = y
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=logits))
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    grad_and_vars = opt.compute_gradients(loss, [w1, w2])
    sparse_grad_and_vars = [0., 0.]

    for i, gv in enumerate(grad_and_vars):
        sparse_grad_and_vars[i] = (tf.multiply(masks[i], gv[0]), gv[1])
    update_sparse = opt.apply_gradients(sparse_grad_and_vars)
    
    other_grad_and_vars = opt.compute_gradients(loss, [b1, b2, b3, w3])
    update_normal = opt.apply_gradients(other_grad_and_vars)

    update = tf.group([update_sparse, update_normal])

    last_layer_grad_and_vars = opt.compute_gradients(loss, [w3, b3])
    last_layer_update = opt.apply_gradients(last_layer_grad_and_vars)
    # the function below is used to obtain original network's accuracy 
    # update = opt.minimize(loss, global_step=global_step)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1)), tf.float32))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    return sess, last_layer_update, update, x, y, accuracy, loss, global_step_increment

if __name__ == "__main__":

    n_batch_size = 1
    n_epochs = 100
    n_batch = int(X.shape[0] / n_batch_size + (X.shape[0] % n_batch_size > 0))

    weights, masks = unsupervised_pretrain()
    sess, last_layer_update, update, x, y, accuracy, loss, global_step_increment = supervised_learn(weights, masks)
    for i in range(n_epochs):
        if i < 5:
            train_acc = 0.
            train_loss = 0.
            for j in range(n_batch):
                if j == n_batch:
                    batch = X[j*n_batch_size::]
                    label_batch = labels[j*n_batch_size:]
                else:
                    batch = X[j*n_batch_size::(j+1)*n_batch_size]
                    label_batch = labels[j*n_batch_size::(j+1)*n_batch_size]

                
                _, acc, batch_loss = sess.run([last_layer_update, accuracy, loss], feed_dict={x:batch, y:label_batch})
                # sess.run(global_step_increment)
                train_acc += acc
                train_loss += batch_loss

        else:
            train_acc = 0.
            train_loss = 0.
            for j in range(n_batch):
                if j == n_batch:
                    batch = X[j*n_batch_size::]
                    label_batch = labels[j*n_batch_size:]
                else:
                    batch = X[j*n_batch_size::(j+1)*n_batch_size]
                    label_batch = labels[j*n_batch_size::(j+1)*n_batch_size]

                
                _, acc, batch_loss = sess.run([update, accuracy, loss], feed_dict={x:batch, y:label_batch})
                # sess.run(global_step_increment)
                train_acc += acc
                train_loss += batch_loss

        acc = sess.run(accuracy, feed_dict={x:X_test, y:labels_test})
        print('epoch-%d:training loss:%f,training accuracy:%f,test accuracy:%f' % (i, train_loss / n_batch, train_acc / n_batch, acc))