from __future__ import print_function
import tensorflow as tf
from data_utils import DEP_RELATION_VOCAB_SIZE
from embedding_utils import VOCAB_SIZE

tf.app.flags.DEFINE_integer('batch_size', 100, 'Training batch size')
tf.app.flags.DEFINE_integer('emb_dim', 200, 'Size of word embeddings')

tf.app.flags.DEFINE_integer('num_kernel', 200,
                            'Number of filters for each window size')

tf.app.flags.DEFINE_integer('min_window', 3, 'Minimum size of filter window')
tf.app.flags.DEFINE_integer('max_window', 5, 'Maximum size of filter window')

tf.app.flags.DEFINE_integer('vocab_size', 40000, 'Vocabulary size')
tf.app.flags.DEFINE_integer('sent_len', 160, 'Input sentence length.')

tf.app.flags.DEFINE_float('l2_reg', 1e-4, 'l2 regularization weight')
tf.app.flags.DEFINE_float('lr', 3e-4, 'l2 learning rate')


class CNNContextModel(object):
    def __init__(self, config):
        self.emb_dim = config['emb_dim']
        self.batch_size = config['batch_size']
        self.num_kernel = config['num_kernel']
        self.min_window = config['min_window']
        self.max_window = config['max_window']
        self.vocab_size = config['vocab_size']
        self.sent_len = config['sent_len']
        self.l2_reg = config['l2_reg']
        self.lr = config['lr']

        '''
        if is_train:
            self.optimizer = config['optimizer']
            self.dropout = config['dropout']
        '''
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)

    def conv2d(self, name, inputs, feature_dim, window_size):
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=self.num_kernel,
                                kernel_size=[window_size, feature_dim],
                                activation=tf.nn.relu,
                                kernel_regularizer=self.regularizer,
                                padding='valid',
                                name=name + '-conv')
        batch_norm = tf.layers.batch_normalization(
            conv,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training)
        conv_len = conv.shape[1]
        pool = tf.layers.max_pooling2d(inputs=batch_norm,
                                       pool_size=[conv_len, 1],
                                       strides=1,
                                       padding='valid',
                                       name=name + '-pool')
        pool_size = self.num_kernel
        flat = tf.reshape(pool, [-1, pool_size])
        return flat

    def build_graph(self):
        self.drop_rate = tf.placeholder(tf.float32)
        self.drop_rate_dense = tf.placeholder(tf.float32)

        self.left_placeholder = tf.placeholder(
            tf.float32, [None, 20, 31], name='left-input')
        self.middle_placeholder = tf.placeholder(
            tf.float32, [None, 80, 31], name='middle-input')
        self.right_placeholder = tf.placeholder(
            tf.float32, [None, 20, 31], name='right-input')
        self.dep_placeholder = tf.placeholder(
            tf.float32, [None, 20, 21], name='dep-input')
        self.path_placeholder = tf.placeholder(
            tf.float32, [None, 20, 22 + DEP_RELATION_VOCAB_SIZE], name='path-input')

        self.is_training = tf.placeholder(tf.bool, name="is-training")

        self.embedding_weights = tf.get_variable(
            name="embedding_weights",
            shape=[VOCAB_SIZE, 200],
            dtype=tf.float32, trainable=False,
            initializer=tf.zeros_initializer())

        left_token, left_feature = tf.split(self.left_placeholder,
                                            [1, 30], 2,
                                            name='left-split')

        middle_token, middle_feature = tf.split(self.middle_placeholder,
                                                [1, 30], 2,
                                                name='middle-split')

        right_token, right_feature = tf.split(self.right_placeholder,
                                              [1, 30], 2,
                                              name='right-split')

        dep_token, dep_feature = tf.split(self.dep_placeholder,
                                          [1, 20], 2,
                                          name='dep-split')

        left_token = tf.cast(tf.reshape(left_token, [-1, 20]),
                             tf.int32,
                             name='left-token')

        middle_token = tf.cast(tf.reshape(middle_token, [-1, 80]),
                               tf.int32,
                               name='middle-token')

        right_token = tf.cast(tf.reshape(right_token, [-1, 20]),
                              tf.int32,
                              name='right-token')

        dep_token = tf.cast(tf.reshape(dep_token, [-1, 20]),
                            tf.int32,
                            name='dep-token')

        left_embed = tf.nn.embedding_lookup(self.embedding_weights,
                                            left_token,
                                            name='left-embed')
        middle_embed = tf.nn.embedding_lookup(self.embedding_weights,
                                              middle_token,
                                              name='middle-embed')
        right_embed = tf.nn.embedding_lookup(self.embedding_weights,
                                             right_token,
                                             name='right-embed')
        dep_embed = tf.nn.embedding_lookup(self.embedding_weights,
                                           dep_token,
                                           name='right-embed')

        left_final = tf.concat([left_embed, left_feature], axis=2, name='left-final')
        left_final = tf.expand_dims(left_final, -1)

        middle_final = tf.concat([middle_embed, middle_feature], axis=2, name='middle-final')
        middle_final = tf.expand_dims(middle_final, -1)

        right_final = tf.concat([right_embed, right_feature], axis=2, name='right-final')
        right_final = tf.expand_dims(right_final, -1)

        dep_token_final = tf.concat([dep_embed, dep_feature], axis=2, name='dep-final')
        dep_token_final = tf.expand_dims(dep_token_final, -1)

        dep_label_final = tf.expand_dims(self.path_placeholder, -1)

        contexts = []
        for name, context_input, length, feature_dim, window_size in [
                ('left', left_final, 20, 230, 3),
                ('middle', middle_final, 20, 230, 3),
                ('right', right_final, 80, 230, 3),
                ('dep_token', dep_token_final, 20, 220, 3),
                ('dep_label', dep_label_final, 20, 22+DEP_RELATION_VOCAB_SIZE, 2)]:
            conved = self.conv2d(name, context_input, feature_dim, window_size)
            contexts.append(conved)

        concat_contexts = tf.concat(contexts, axis=1, name='concat')

        dropout = tf.layers.dropout(concat_contexts, self.drop_rate,
                                    training=self.is_training, name='dropout-combined')

        dense1 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu,
                                 kernel_regularizer=self.regularizer, name='dense-1')
        dense1_batch = tf.layers.batch_normalization(
            dense1,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training)
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
            100,  # Decay step.
            0.9,  # Decay rate.
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)
