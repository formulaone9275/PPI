from __future__ import print_function
import tensorflow as tf
import numpy as np
from constant import *
from data_utils_new import load_sentence_matrix, batch_iter, create_tensor
from lstm import LSTMModel
from cnn_yifan import CNNModel
from embedding_utils import VOCAB_SIZE, EMBEDDING
from build_tfrecord_file import iter_dataset
from flags import *


class Train(object):
    def __init__(self, model_class):
        self.model_class = model_class
        FLAGS = tf.flags.FLAGS
        self.batch_size = FLAGS.batch_size
        self.epoch = FLAGS.epoch
        self.log_dir = FLAGS.log_dir
        
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
        #eval_sent, eval_head, eval_dep = eval_data
        #sent_mx, sent_len = create_tensor(*eval_sent)
        #head_mx, head_len = create_tensor(*eval_head)
        #dep_mx, dep_len = create_tensor(*eval_dep)

        data = iter_dataset(self.sess, eval_data, 1, 10000)
        sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, labels = next(data)
        
        p, r, f, l, s = self.sess.run(
            [self.model.precision, self.model.recall,
             self.model.fscore, self.model.loss, self.summary],
            feed_dict={
                self.model.ph_sent: sent_mx,
                self.model.ph_sent_len: sent_len,
                self.model.ph_head: head_mx,
                self.model.ph_head_len: head_len,
                self.model.ph_dep: dep_mx,
                self.model.ph_dep_len: dep_len,
                self.model.label_placeholder: labels,
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

    def train(self, train_data, eval_sets):
        with tf.Graph().as_default():
            model = self.model_class()
            model.build_graph()
            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            self.model = model

            with tf.Session() as sess:
                self.sess = sess
                self.add_summary()
                sess.run([init, init_l])
                sess.run(tf.assign(model.embedding_weights, EMBEDDING))
                losses = []

                '''
                for e, i, step, batch_data, batch_labels in batch_iter(
                        train_data, train_labels, self.epoch, self.batch_size):
                    batch_sent, batch_head, batch_dep = batch_data
                    batch_sent_mx, batch_sent_len = batch_sent
                    batch_head_mx, batch_head_len = batch_head
                    batch_dep_mx, batch_dep_len = batch_dep
                '''
                step = 0
                for batch_data in iter_dataset(sess, train_data, self.epoch, self.batch_size):
                    batch_sent_mx, batch_sent_len, batch_head_mx, batch_head_len, batch_dep_mx, batch_dep_len, batch_labels = batch_data

                    _, mini_loss = sess.run(
                        [model.train_op, model.loss],
                        feed_dict={
                            model.ph_sent: batch_sent_mx,
                            model.ph_sent_len: batch_sent_len,
                            model.ph_head: batch_head_mx,
                            model.ph_head_len: batch_head_len,
                            model.ph_dep: batch_dep_mx,
                            model.ph_dep_len: batch_dep_len,
                            model.label_placeholder: batch_labels,
                            model.drop_rate: 0.5,
                            model.drop_rate_dense: 0.2,
                            model.is_training: True,
                        })
                    losses.append(mini_loss)
                    
                    if DEBUG or (step > 0 and step % 500 == 0):
                        '''
                        print('\nstep {}, epoch {}, batch {}, loss {}'.format(
                            step, e, i, np.mean(losses)))
                        
                        for name, eval_data, eval_labels in eval_sets:
                            self.eval(name, eval_data, eval_labels, step)
                        '''
                        print('\nstep {}, loss {}'.format(
                            step, np.mean(losses)))
                        
                        for name, eval_data in eval_sets:
                            self.eval(name, eval_data, step)
                        losses = []

                    if DEBUG:
                        break
                    
                    step += 1

if __name__ == '__main__':
    DATASETS = {
        'ppi': ('ppi_train.txt', 'ppi_train_dev.txt', 'ppi_dev.txt', 'aimed_dev.txt'),
        'mirtar': ('mirtar_train.txt', 'mirtar_train_dev.txt', 'mirtar_dev.txt', 'mirtex_dev.txt'),
    }

    DATASETS = {k: ['data/' + vv for vv in v] for k, v in DATASETS.items()}

    RELATION = 'ppi'

    # I can't find a way to load data from disk quickly.
    #train_data, train_labels = load_sentence_matrix(DATASETS[RELATION][0])
    #tdev_data, tdev_labels = load_sentence_matrix(DATASETS[RELATION][1])
    #dev_data, dev_labels = load_sentence_matrix(DATASETS[RELATION][2])
    #test_data, test_labels = load_sentence_matrix(DATASETS[RELATION][3])

    '''
    eval_sets = [('train_dev', tdev_data, tdev_labels),
                 ('dev', dev_data, dev_labels),
                 ('test', test_data, test_labels)]
    '''
    
    FLAGS = tf.flags.FLAGS
    if FLAGS.model == 'cnn':
        model = CNNModel
    elif FLAGS.model == 'lstm':
        model = LSTMModel

    train = Train(model)
    #train.train(train_data, train_labels, eval_sets)
    train.train('data/aimed_training.tfrecords',
                [#('train_dev', 'data/ppi_train_dev.tfrecords'),
                 #('dev', 'data/ppi_dev.tfrecords'),
                 ('dev', 'data/aimed_training.tfrecords')])
                 
