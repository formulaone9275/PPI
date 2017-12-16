from __future__ import print_function
import tensorflow as tf
import numpy as np
from constant import *
from data_utils_new import load_sentence_matrix
from data_utils_new import DEP_RELATION_VOCAB_SIZE
from embedding_utils import VOCAB_SIZE, EMBEDDING
from flags import *


class CNNModel(object):
    def __init__(self):
        FLAGS = tf.flags.FLAGS
        self.emb_dim = FLAGS.emb_dim
        self.batch_size = FLAGS.batch_size
        self.epoch = FLAGS.epoch
        self.num_kernel = FLAGS.num_kernel
        self.min_window = FLAGS.min_window
        self.max_window = FLAGS.max_window
        self.l2_reg = FLAGS.l2_reg
        self.lr = FLAGS.lr
        self.lr_decay_step = FLAGS.decay_step
        self.lr_decay_rate = FLAGS.decay_rate
        self.use_head = FLAGS.use_head
        self.use_dep = FLAGS.use_dep
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)

    def conv2d(self, name, inputs, window_size):
        feature_dim = int(inputs.shape[-2])
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=self.num_kernel,
                                kernel_size=[window_size, feature_dim],
                                activation=tf.nn.relu,
                                kernel_regularizer=self.regularizer,
                                padding='valid',
                                name=name + '-conv')
        '''
        batch_norm = tf.layers.batch_normalization(
            conv,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training,
            name=name+'-batch-norm')
        '''
        conv_len = int(conv.shape[1])
        pool = tf.layers.max_pooling2d(inputs=conv,
                                       pool_size=[conv_len, 1],
                                       strides=1,
                                       padding='valid',
                                       name=name + '-pool')
        pool_size = self.num_kernel
        flat = tf.reshape(pool, [-1, pool_size], name=name+"-flat")
        return flat

    def build_graph(self):
        self.drop_rate = tf.placeholder(tf.float32)
        self.drop_rate_dense = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name="is-training")

        with tf.device("cpu:0"):
            self.embedding_weights = tf.get_variable(
                name="embedding_weights",
                shape=[VOCAB_SIZE, 200],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            self.ph_sent = tf.placeholder(tf.float32, [None, 160, 155], name='sent-input')
            self.ph_sent_len = tf.placeholder(tf.int32, [None,], name='sent-len-input')
        
            self.ph_head = tf.placeholder(tf.float32, [None, 160, 155], name='head-input')
            self.ph_head_len = tf.placeholder(tf.int32, [None,], name='head-len-input')
        
            self.ph_dep = tf.placeholder(tf.float32, [None, 20, 155], name='dep-input')
            self.ph_dep_len = tf.placeholder(tf.int32, [None,], name='dep-len-input')
        
            inputs = [('sent', self.ph_sent, self.ph_sent_len)]
            if self.use_head:
                inputs.append(('head', self.ph_head, self.ph_head_len))
                print("Use head tokens")
            if self.use_dep:
                inputs.append(('dep', self.ph_dep, self.ph_dep_len))
                print("Use dep tokens")

            input_vecs = []
            for name, seq, _ in inputs:
                seq_len = int(seq.shape[1])
                token, feature = tf.split(seq, [1, 154], 2, name=name+'-split')
                token = tf.cast(tf.reshape(token, [-1, seq_len]), tf.int32, name=name+'-token')
                embed = tf.nn.embedding_lookup(self.embedding_weights, token, name=name+'-embed')
                final = tf.concat([embed, feature], axis=2, name=name+'-final')
                final = tf.expand_dims(final, -1)
                input_vecs.append((name, final, 3))

        pools = []
        for name, context_input, window_size in input_vecs:
            conved = self.conv2d(name, context_input, window_size)
            pools.append(conved)
            
        concat_contexts = tf.concat(pools, axis=1, name='combined')
        dropout = tf.layers.dropout(concat_contexts, self.drop_rate,
                                    training=self.is_training, name='dropout-combined')

        dense1 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu,
                                 kernel_regularizer=self.regularizer, name='dense-1')
        '''
        dense1_batch = tf.layers.batch_normalization(
            dense1,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training, name='batch_norm_dense1')
        '''
        dropout1 = tf.layers.dropout(dense1, self.drop_rate_dense,
                                     training=self.is_training, name='dropout-1')
        '''
        dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu,
                                kernel_regularizer=self.regularizer, name='dense-2')
        dense2_batch = tf.layers.batch_normalization(
            dense2,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training)
        dropout2 = tf.layers.dropout(dense2_batch, self.drop_rate_dense,
                                     training=self.is_training, name='dropout-2')
        '''
        
        logits = tf.layers.dense(inputs=dropout1, units=2,
                                 kernel_regularizer=self.regularizer, name='output')

        self.label_placeholder = tf.placeholder(tf.int32, [None, 2])

        all_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=self.label_placeholder,
                                                       name='losses')
        self.loss = tf.reduce_mean(all_loss, name='batch-loss')

        pred = tf.argmax(tf.nn.softmax(logits), 1)
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

        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.learning_rate = tf.train.exponential_decay(
            self.lr,  # Base learning rate.
            global_step,  # Current index into the dataset.
            self.lr_decay_step,  # Decay step.
            self.lr_decay_rate,  # Decay rate.
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)
