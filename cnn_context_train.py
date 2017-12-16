from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from constant import *
from data_utils import load_context_matrix
from embedding_utils import EMBEDDING
from cnn_context_model import CNNContextModel


class Train(object):
    def __init__(self, config):
        # DNN.
        #config = tf.ConfigProto(device_count={'CPU': 36})
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--log_dir',
            type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                 'tensorflow/mnist/logs/mnist_with_summaries'),

            help='Summaries log directory')
        FLAGS, unparsed = parser.parse_known_args()
        '''
        self.config = config
        self.log_dir = config['log_dir']
        
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

    def eval(self, name, eval_data, step):
        eval_left, eval_middle, eval_right, eval_mx, eval_path, eval_label = eval_data
        p, r, f, l, s = self.sess.run(
            [self.model.precision, self.model.recall,
             self.model.fscore, self.model.loss, self.summary],
            feed_dict={
                self.model.left_placeholder: eval_left,
                self.model.middle_placeholder: eval_middle,
                self.model.right_placeholder: eval_right,
                self.model.dep_placeholder: eval_mx,
                self.model.path_placeholder: eval_path,
                self.model.label_placeholder: eval_label,
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

    @staticmethod
    def batch_iter(data, epoch_num, batch_size):
        left, middle, right, dep_mx, path_mx, labels = data
        data_size = left.shape[0]
        batch_num = data_size / batch_size

        for e in range(epoch_num):
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_left = left[shuffle_indices]
            shuffled_middle = middle[shuffle_indices]
            shuffled_right = right[shuffle_indices]
            shuffled_mx = dep_mx[shuffle_indices]
            shuffled_path = path_mx[shuffle_indices]
            shuffled_labels = labels[shuffle_indices]

            for i in range(batch_num):
                batch_left = shuffled_left[i * batch_size:(i + 1) * batch_size]
                batch_middle = shuffled_middle[i * batch_size:(i + 1) * batch_size]
                batch_right = shuffled_right[i * batch_size:(i + 1) * batch_size]
                batch_mx = shuffled_mx[i * batch_size:(i + 1) * batch_size]
                batch_path = shuffled_path[i * batch_size:(i + 1) * batch_size]
                batch_labels = shuffled_labels[i * batch_size:(i + 1) * batch_size]
                batch_step = e * batch_num + i
                yield e, i, batch_step, \
                    (batch_left, batch_middle, batch_right, batch_mx, batch_path, batch_labels)
    
    def train(self, train_data, eval_data):
        batch_size = 256

        with tf.Graph().as_default():
            model = CNNContextModel(self.config)
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
                for e, i, step, batch in self.batch_iter(train_data, 10, batch_size):
                    batch_left, batch_middle, batch_right, batch_mx, batch_path, batch_labels = batch
                    _, mini_loss = sess.run([model.train_op, model.loss], feed_dict={
                        model.left_placeholder: batch_left,
                        model.middle_placeholder: batch_middle,
                        model.right_placeholder: batch_right,
                        model.dep_placeholder: batch_mx,
                        model.path_placeholder: batch_path,
                        model.label_placeholder: batch_labels,
                        model.drop_rate: 0.2,
                        model.drop_rate_dense: 0.2,
                        model.is_training: True,
                    })
                    losses.append(mini_loss)

                    if i % 100 == 0:
                        print('\nepoch {}, batch {}, loss {}'.format(e, i, np.mean(losses)))
                        for name, eval_set in eval_data:
                            self.eval(name, eval_set, step)
                        losses = []
                        
                    if step > 0 and step % 2000 == 0:
                        global_step = sess.run(model.global_step)
                        path = saver.save(
                            sess, "checkpoints/cnn_context", global_step=global_step)
                        
                    if DEBUG:
                        break


DATASETS = {
    'ppi': ('ppi_train.txt', 'ppi_train_dev.txt', 'ppi_dev.txt', 'aimed_dev.txt'),
    'mirtar': ('mirtar_train.txt', 'mirtar_train_dev.txt', 'mirtar_dev.txt', 'mirtex_dev.txt'),
}

DATASETS = {k: ['data/' + vv for vv in v] for k, v in DATASETS.items()}

RELATION = 'ppi'

if DEBUG:
    train_data = load_context_matrix(DATASETS[RELATION][0])
else:
    train_set = np.load('data/' + RELATION + '_train_cont.npz')
    train_data = (train_set['left'], train_set['middle'],
                  train_set['right'], train_set['dep_matrix'],
                  train_set['path_matrix'], train_set['labels'])

tdev_data = load_context_matrix(DATASETS[RELATION][1])
dev_data = load_context_matrix(DATASETS[RELATION][2])
test_data = load_context_matrix(DATASETS[RELATION][3])

eval_data = [('train_dev', tdev_data),
             ('dev', dev_data),
             ('test', test_data)]

tf.app.flags.DEFINE_string('log_dir', 'logs', 'log dir')
FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()
config = dict(FLAGS.__flags.items())
train = Train(config)
train.train(train_data, eval_data)
