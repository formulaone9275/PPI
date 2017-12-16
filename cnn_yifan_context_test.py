from __future__ import print_function
import tensorflow as tf
import numpy as np
from constant import *
from data_utils_new import load_context_matrix, load_tagged
from cnn_context_model import CNNContextModel
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pickle

def evaluate(eval_data, tagged_sents, config):
    with tf.Graph().as_default():
        model = CNNContextModel(config)
        model.build_graph()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('checkpoints')
            saver.restore(sess, ckpt.model_checkpoint_path)
            left, middle, right, dep, label = eval_data
            feed_dict = {
                model.left_placeholder: left,
                model.middle_placeholder: middle,
                model.right_placeholder: right,
                model.dep_placeholder: dep,
                model.label_placeholder: label,
                model.drop_rate: 0,
                model.drop_rate_dense: 0,
                model.is_training: False,
            }
            p, r, f, prob = sess.run(
                [model.precision, model.recall, model.fscore, model.prob],
                feed_dict=feed_dict)
            print(p,r,f)
            
            pos_prob = []
            pos_bin = []
            precision = dict()
            recall = dict()
            average_precision = dict()
            thresholds = dict()
            for pr, sent, l in zip(prob, tagged_sents, label):
                pos_prob.append(pr[1])
                pos_bin.append(l[1])
                if (pr[1] < 0.5 and l[1] == 1):
                    print('FN', str(pr[1]), str(l[1]), sent)
                if (pr[1] > 0.5 and l[1] == 0):
                    print('FP', str(pr[1]), str(l[1]), sent)

            precision[0], recall[0], thresholds[0] = precision_recall_curve(
                pos_bin, pos_prob)
            average_precision[0] = average_precision_score(pos_bin, pos_prob)
            with open('pk/cnn_yifan_context.pk', 'wb') as f:
                pickle.dump((precision, recall, average_precision, thresholds), f)
                
if __name__ == '__main__':
    tf.app.flags.DEFINE_string('log_dir', 'logs', 'log dir')
    FLAGS = tf.app.flags.FLAGS
    FLAGS._parse_flags()
    config = dict(FLAGS.__flags.items())
    config['use_dep'] = True

    test_data = load_context_matrix('data2/aimed_train.txt')
    tagged_sents = load_tagged('data2/aimed_train.txt')
    evaluate(test_data, tagged_sents, config)
