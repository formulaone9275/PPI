from __future__ import print_function
import tensorflow as tf
import numpy as np
from constant import *
from data_utils_new import load_context_matrix, batch_iter, create_tensor
from data_utils_new import DEP_RELATION_VOCAB_SIZE
from embedding_utils import VOCAB_SIZE, EMBEDDING
from flags import *


class CNNContextModel(object):
    def __init__(self):
        FLAGS = tf.flags.FLAGS
        self.emb_dim = FLAGS.emb_dim
        self.num_kernel = FLAGS.num_kernel
        self.min_window = FLAGS.min_window
        self.max_window = FLAGS.max_window
        self.l2_reg = FLAGS.l2_reg
        self.lr = FLAGS.lr
        self.lr_decay_step = FLAGS.decay_step
        self.lr_decay_rate = FLAGS.decay_rate
        self.use_dep = FLAGS.use_dep
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)

    def conv2d(self, name, inputs, feature_dim, window_stride):
        flats = []
        for win_size, strides in window_stride:
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=self.num_kernel,
                                    kernel_size=[win_size, feature_dim],
                                    strides=strides,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=self.regularizer,
                                    padding='valid',
                                    name='conv-{}-{}'.format(name, win_size))

            batch_norm = tf.layers.batch_normalization(
                conv,
                beta_regularizer=self.regularizer,
                gamma_regularizer=self.regularizer,
                training=self.is_training,
                name='batch-norm-{}-{}'.format(name, win_size))

            conv_len = int(conv.shape[1])
            pool = tf.layers.max_pooling2d(inputs=batch_norm,
                                           pool_size=[conv_len, 1],
                                           strides=1,
                                           padding='valid',
                                           name='pool-{}-{}'.format(name, win_size))
            pool_size = self.num_kernel
            flats.append(tf.reshape(pool, [-1, pool_size], name='flat-{}-{}'.format(name, win_size)))
        return flats

    def build_graph(self):
        self.drop_rate = tf.placeholder(tf.float32)
        self.drop_rate_dense = tf.placeholder(tf.float32)

        input_vecs = []
        self.is_training = tf.placeholder(tf.bool, name="is-training")

        self.embedding_weights = tf.get_variable(
            name="embedding_weights",
            shape=[VOCAB_SIZE, 200],
            dtype=tf.float32, trainable=False,
            initializer=tf.zeros_initializer())

        self.left_placeholder = tf.placeholder(tf.float32, [None, 20, 155], name='left-input')
        self.middle_placeholder = tf.placeholder(tf.float32, [None, 80, 155], name='middle-input')
        self.right_placeholder = tf.placeholder(tf.float32, [None, 20, 155], name='right-input')
        self.dep_placeholder = tf.placeholder(tf.float32, [None, 20, 155], name='dep-input')
        
        for name, placeholder, window_stride in [
                ('left', self.left_placeholder, [(3, 1),]),
                ('middle', self.middle_placeholder, [(3, 1), ]),
                ('right', self.right_placeholder, [(3, 1), ]),
                ('dep', self.dep_placeholder, [(2, 1), ]),
        ]:
            if not self.use_dep and name == 'dep':
                continue
            length = int(placeholder.shape[1])
            input_token, input_feature = tf.split(placeholder, [1, 154], 2, name=name+'-split')
            input_token = tf.cast(tf.reshape(input_token, [-1, length]), tf.int32, name=name+'-token')
            input_embed = tf.nn.embedding_lookup(self.embedding_weights, input_token, name=name+'-embed')
            input_final = tf.concat([input_embed, input_feature], axis=2, name=name+'-final')
            input_final = tf.expand_dims(input_final, -1)
            input_vecs.append((name, input_final, length, 354, window_stride))

        pools = []
        for name, context_input, length, feature_dim, window_size in input_vecs:
            conveds = self.conv2d(name, context_input, feature_dim, window_size)
            pools += conveds
            
        concat_contexts = tf.concat(pools, axis=1, name='combined')
        dropout = tf.layers.dropout(concat_contexts, self.drop_rate,
                                    training=self.is_training, name='dropout-combined')

        dense1 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu,
                                 kernel_regularizer=self.regularizer, name='dense-1')

        dense1_batch = tf.layers.batch_normalization(
            dense1,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training, name='batch_norm_dense1')

        dropout1 = tf.layers.dropout(dense1_batch, self.drop_rate_dense,
                                     training=self.is_training, name='dropout-1')

        dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu,
                                kernel_regularizer=self.regularizer, name='dense-2')

        dense2_batch = tf.layers.batch_normalization(
            dense2,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training)

        dropout2 = tf.layers.dropout(dense2_batch, self.drop_rate_dense,
                                     training=self.is_training, name='dropout-2')

        logits = tf.layers.dense(inputs=dropout2, units=2,
                                 kernel_regularizer=self.regularizer, name='output')

        self.label_placeholder = tf.placeholder(tf.int32, [None, 2])

        all_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=self.label_placeholder,
                                                       name='losses')
        self.loss = tf.reduce_mean(all_loss, name='batch-loss')

        pred = tf.argmax(tf.nn.softmax(logits), 1)
        self.prob = tf.nn.softmax(logits)
        gold = tf.argmax(self.label_placeholder, 1)
        tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(gold, tf.bool))
        fp = tf.logical_and(tf.cast(pred, tf.bool),
                            tf.logical_not(tf.cast(gold, tf.bool)))
        fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)),
                            tf.cast(gold, tf.bool))
        self.precision = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                               tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)),
                               name='precision')
        self.recall = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                            tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)),
                            name='recall')

        self.fscore = self.precision * self.recall * 2 / (self.precision + self.recall)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.learning_rate = tf.train.exponential_decay(
            self.lr,  # Base learning rate.
            self.global_step,  # Current index into the dataset.
            self.lr_decay_step,  # Decay step.
            self.lr_decay_rate,  # Decay rate.
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


class Train(object):
    def __init__(self):
        FLAGS = tf.app.flags.FLAGS
        self.log_dir = FLAGS.log_dir
        self.batch_size = FLAGS.batch_size
        self.epoch = FLAGS.epoch
        
    def add_summary(self):
        tf.summary.scalar('precision', self.model.precision)
        tf.summary.scalar('recall', self.model.recall)
        tf.summary.scalar('fscore', self.model.fscore)
        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('learning_rate', self.model.learning_rate)
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.train_dev_writer = tf.summary.FileWriter(self.log_dir + '/train_dev', self.sess.graph)
        self.dev_writer = tf.summary.FileWriter(self.log_dir + '/dev', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)

    def eval(self, name, eval_data, eval_labels, step):
        eval_left, eval_middle, eval_right, eval_dep = eval_data
        left_mx, left_len = create_tensor(*eval_left)
        middle_mx, middle_len = create_tensor(*eval_middle)
        right_mx, right_len = create_tensor(*eval_right)
        dep_mx, dep_len = create_tensor(*eval_dep)

        p, r, f, l, s = self.sess.run(
            [self.model.precision, self.model.recall,
             self.model.fscore, self.model.loss, self.summary],
            feed_dict={
                self.model.left_placeholder: left_mx,
                self.model.middle_placeholder: middle_mx,
                self.model.right_placeholder: right_mx,
                self.model.dep_placeholder: dep_mx,
                self.model.label_placeholder: eval_labels,
                self.model.drop_rate: 0,
                self.model.drop_rate_dense: 0,
                self.model.is_training: False,
            })
        if name == 'train':
            self.train_writer.add_summary(s, step)
        elif name == 'train_dev':
            self.train_dev_writer.add_summary(s, step)
        elif name == 'dev':
            self.dev_writer.add_summary(s, step)
        elif name == 'test':
            self.test_writer.add_summary(s, step)
        print('{}: prec {}, recall {}, fscore {}, loss {}'.format(
            name, p, r, f, l))

    def train(self, train_data, train_labels, eval_data):
        with tf.Graph().as_default():
            model = CNNContextModel()
            model.build_graph()
            saver = tf.train.Saver(tf.global_variables())
            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            self.model = model

            with tf.Session() as sess:
                self.sess = sess
                self.add_summary()
                sess.run(init)
                sess.run(init_l)
                sess.run(tf.assign(model.embedding_weights, EMBEDDING))
                losses = []
                for e, i, step, batch_data, batch_labels in batch_iter(
                        train_data, train_labels, self.epoch, self.batch_size):
                    batch_left, batch_middle, batch_right, batch_dep = batch_data
                    batch_left_mx, batch_left_len = batch_left
                    batch_middle_mx, batch_middle_len = batch_middle
                    batch_right_mx, batch_right_len = batch_right
                    batch_dep_mx, batch_dep_len = batch_dep
                    
                    _, mini_loss = sess.run(
                        [model.train_op, model.loss],
                        feed_dict={
                            model.left_placeholder: batch_left_mx,
                            model.middle_placeholder: batch_middle_mx,
                            model.right_placeholder: batch_right_mx,
                            model.dep_placeholder: batch_dep_mx,
                            model.label_placeholder: batch_labels,
                            model.drop_rate: 0.5,
                            model.drop_rate_dense: 0.2,
                            model.is_training: True,
                        })
                    losses.append(mini_loss)

                    if DEBUG or (step > 0 and step % 500 == 0):
                        print('\nstep {}, epoch {}, batch {}, loss {}'.format(
                            step, e, i, np.mean(losses)))
                        for name, eval_set, eval_labels in eval_data:
                            self.eval(name, eval_set, eval_labels, step)
                        losses = []

                    if SAVE_MODEL and (step > 0 and step % 4000 == 0):
                        global_step = sess.run(model.global_step)
                        path = saver.save(
                            sess, "checkpoints/yifan_context", global_step=global_step)

                    if DEBUG:
                        break


if __name__ == '__main__':
    DATASETS = {
        'ppi': ('ppi_train.txt',
                'ppi_train_dev.txt',
                'ppi_dev.txt',
                'aimed_dev.txt'),
        'mirtar': ('mirtar_train.txt',
                   'mirtar_train_dev.txt',
                   'mirtar_dev.txt',
                   'mirtex_dev.txt'),
    }

    DATASETS = {k: ['data/' + vv for vv in v] for k, v in DATASETS.items()}

    RELATION = 'ppi'

    train_data, train_labels = load_context_matrix(DATASETS[RELATION][0])
    tdev_data, tdev_labels = load_context_matrix(DATASETS[RELATION][1])
    dev_data, dev_labels = load_context_matrix(DATASETS[RELATION][2])
    test_data, test_labels = load_context_matrix(DATASETS[RELATION][3])

    eval_data = [('train_dev', tdev_data, tdev_labels),
                 ('dev', dev_data, dev_labels),
                 ('test', test_data, test_labels)]

    train = Train()
    train.train(train_data, train_labels, eval_data)
