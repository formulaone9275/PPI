from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from constant import *
from data_utils import load_context_matrix, DEP_RELATION_VOCAB_SIZE
from embedding_utils import EMBEDDING, VOCAB_SIZE


DATASETS = {
    'ppi': ('ppi_train.txt', 'ppi_train_dev.txt', 'ppi_dev.txt', 'aimed_dev.txt'),
    'mirtar': ('mirtar_train.txt', 'mirtar_train_dev.txt', 'mirtar_dev.txt', 'mirtex_dev.txt'),
}

DATASETS = {k: ['data/'+vv for vv in v] for k, v in DATASETS.items()}

RELATION = 'ppi'

if DEBUG:
    train_left, train_middle, train_right, \
        train_dep_mx, train_path_mx, train_labels = load_context_matrix(DATASETS[RELATION][0])
else:
    train_set = np.load('data/'+RELATION+'_train_cont.npz')
    train_left, train_middle, train_right, train_dep_mx, train_path_mx, train_labels = \
        (train_set['left'], train_set['middle'],
         train_set['right'], train_set['dep_matrix'], train_set['path_matrix'], train_set['labels'])

tdev_left, tdev_middle, tdev_right, \
    tdev_dep_mx, tdev_path_mx, tdev_labels = load_context_matrix(DATASETS[RELATION][1])
dev_left, dev_middle, dev_right, \
    dev_dep_mx, dev_path_mx, dev_labels = load_context_matrix(DATASETS[RELATION][2])
test_left, test_middle, test_right, \
    test_dep_mx, test_path_mx, test_labels = load_context_matrix(DATASETS[RELATION][3])

# DNN.

regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
keep_prob = tf.placeholder(tf.float32)
keep_prob_dense = tf.placeholder(tf.float32)

left_placeholder = tf.placeholder(tf.float32, [None, 20, 31], name='left-input')
middle_placeholder = tf.placeholder(tf.float32, [None, 80, 31], name='middle-input')
right_placeholder = tf.placeholder(tf.float32, [None, 20, 31], name='right-input')
dep_placeholder = tf.placeholder(tf.float32, [None, 20, 21], name='dep-input')
path_placeholder = tf.placeholder(tf.float32, [None, 20, 22+DEP_RELATION_VOCAB_SIZE], name='path-input')
training_placeholder = tf.placeholder(tf.bool, name='is-training')

#path_embedding_weights = tf.get_variable(
#    name="path_embedding_weights",
#    shape=[DEP_RELATION_VOCAB_SIZE-1, 20],
#    dtype=tf.float32, trainable=True,
#    initializer=tf.truncated_normal_initializer())

embedding_weights = tf.get_variable(name="embedding_weights", shape=[VOCAB_SIZE, 200],
                                    dtype=tf.float32, trainable=False,
                                    initializer=tf.zeros_initializer())

left_token, left_feature = tf.split(left_placeholder, [1, 30], 2, name='left-split')
middle_token, middle_feature = tf.split(middle_placeholder, [1, 30], 2, name='middle-split')
right_token, right_feature = tf.split(right_placeholder, [1, 30], 2, name='right-split')
dep_token, dep_feature = tf.split(dep_placeholder, [1, 20], 2, name='dep-split')
#path_token, path_feature = tf.split(path_placeholder, [1, 22], 2, name='path-split')

left_token = tf.cast(tf.reshape(left_token, [-1, 20]), tf.int32, name='left-token')
middle_token = tf.cast(tf.reshape(middle_token, [-1, 80]), tf.int32, name='middle-token')
right_token = tf.cast(tf.reshape(right_token, [-1, 20]), tf.int32, name='right-token')
dep_token = tf.cast(tf.reshape(dep_token, [-1, 20]), tf.int32, name='dep-token')
#path_token = tf.cast(tf.reshape(path_token, [-1, 20]), tf.int32, name='path-token')

left_embed = tf.nn.embedding_lookup(embedding_weights, left_token, name='left-embed')
middle_embed = tf.nn.embedding_lookup(embedding_weights, middle_token, name='middle-embed')
right_embed = tf.nn.embedding_lookup(embedding_weights, right_token, name='right-embed')
dep_embed = tf.nn.embedding_lookup(embedding_weights, dep_token, name='right-embed')
#path_embed = tf.nn.embedding_lookup(path_embedding_weights, path_token, name='path-embed')

left_final = tf.concat([left_embed, left_feature], axis=2, name='left-final')
left_final = tf.expand_dims(left_final, -1)

middle_final = tf.concat([middle_embed, middle_feature], axis=2, name='middle-final')
middle_final = tf.expand_dims(middle_final, -1)

right_final = tf.concat([right_embed, right_feature], axis=2, name='right-final')
right_final = tf.expand_dims(right_final, -1)

dep_final = tf.concat([dep_embed, dep_feature], axis=2, name='dep-final')
dep_final = tf.expand_dims(dep_final, -1)

path_final = tf.expand_dims(path_placeholder, -1)

left_conv = tf.layers.conv2d(inputs=left_final,
                             filters=100,
                             kernel_size=[3, 230],
                             activation=tf.nn.relu,
                             kernel_regularizer=regularizer,
                             padding='valid',
                             name='left-conv')

#left_batch = tf.layers.batch_normalization(left_conv, training=training_placeholder)
                                           
left_pool = tf.layers.max_pooling2d(inputs=left_conv, pool_size=[18,1],
                                    strides=[1, 1], padding='valid',
                                    name='left-pool')

left_flat = tf.reshape(left_pool, [-1, 100], name='left-flat')

middle_conv = tf.layers.conv2d(inputs=middle_final,
                               filters=200,
                               kernel_size=[3, 230],
                               activation=tf.nn.relu,
                               kernel_regularizer=regularizer,
                               padding='valid',
                               name='middle-conv')
#middle_batch = tf.layers.batch_normalization(middle_conv, training=training_placeholder)
middle_pool = tf.layers.max_pooling2d(inputs=middle_conv, pool_size=[78,1],
                                      strides=[1, 1], padding='valid', name='middle-pool')

middle_flat = tf.reshape(middle_pool, [-1, 200], name='middle-flat')

right_conv = tf.layers.conv2d(inputs=right_final,
                              filters=100,
                              kernel_size=[3, 230],
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              padding='valid',
                              name='right-conv')
#right_batch = tf.layers.batch_normalization(right_conv, training=training_placeholder)
right_pool = tf.layers.max_pooling2d(inputs=right_conv, pool_size=[18,1],
                                     strides=[1, 1], padding='valid', name='right-pool')

right_flat = tf.reshape(right_pool, [-1, 100], name='right-flat')

dep_conv = tf.layers.conv2d(inputs=dep_final,
                            filters=100,
                            kernel_size=[3, 220],
                            activation=tf.nn.relu,
                            kernel_regularizer=regularizer,
                            padding='valid',
                            name='dep-conv')
#dep_batch = tf.layers.batch_normalization(dep_conv, training=training_placeholder)
dep_pool = tf.layers.max_pooling2d(inputs=dep_conv, pool_size=[18,1],
                                   strides=[1, 1], padding='valid', name='dep-pool')

dep_flat = tf.reshape(dep_pool, [-1, 100], name='dep-flat')

path_conv = tf.layers.conv2d(inputs=path_final,
                             filters=50,
                             kernel_size=[2, 22+DEP_RELATION_VOCAB_SIZE],
                             activation=tf.nn.relu,
                             kernel_regularizer=regularizer,
                             padding='valid',
                             name='path-conv')
#path_batch = tf.layers.batch_normalization(path_conv, training=training_placeholder)
path_pool = tf.layers.max_pooling2d(inputs=path_conv, pool_size=[19,1],
                                   strides=[1, 1], padding='valid', name='path-pool')

path_flat = tf.reshape(path_pool, [-1, 50], name='path-flat')

combined = tf.concat([left_flat, middle_flat, right_flat, dep_flat, path_flat],
                     axis=1, name='combined')

dropout = tf.layers.dropout(combined, keep_prob, name='dropout-combined', training=training_placeholder)
'''
com_drop = tf.expand_dims(dropout, -1)
conv2 = tf.layers.conv1d(inputs=com_drop,
                         filters=400,
                         kernel_size=10,
                         strides=5,
                         kernel_regularizer=regularizer,
                         activation=tf.nn.relu)
print(conv2.shape)

conv2_batch = tf.layers.batch_normalization(conv2, training=training_placeholder)

conv2_pool = tf.layers.max_pooling1d(inputs=conv2_batch, pool_size=179,
                                     strides=1, padding='valid', name='conv2-pool')
print(conv2_pool.shape)
dropout0 = tf.layers.dropout(conv2_pool, keep_prob, name='dropout-combined-2')
conv2_flat = tf.reshape(dropout0, [-1, 400], name='conv2-flat')
'''
dense1 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu,
                         kernel_regularizer=regularizer, name='dense-1')
dense1_batch = tf.layers.batch_normalization(dense1, training=training_placeholder)
dropout1 = tf.layers.dropout(dense1, keep_prob_dense, name='dropout-1', training=training_placeholder)

dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu,
                         kernel_regularizer=regularizer, name='dense-2')
#dense2_batch = tf.layers.batch_normalization(dense2, training=training_placeholder)
dropout2 = tf.layers.dropout(dense2, keep_prob_dense, name='dropout-2')

logits = tf.layers.dense(inputs=dropout2, units=2,
                         kernel_regularizer=regularizer, name='output')

label_placeholder = tf.placeholder(tf.int32, [None, 2])

loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_placeholder, name='losses')
batch_loss = tf.reduce_mean(loss, name='batch-loss')

pred = tf.argmax(tf.nn.softmax(logits), 1)
gold = tf.argmax(label_placeholder, 1)
tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(gold, tf.bool))
fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(gold, tf.bool)))
fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(gold, tf.bool))
precision = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)), name='precision')
recall = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)), name='recall')

fscore = precision * recall * 2 / (precision+recall)

global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(
    0.001,                # Base learning rate.
    global_step,  # Current index into the dataset.
    200,          # Decay step.
    0.90,         # Decay rate.
    staircase=True
)

optimizer = tf.train.AdamOptimizer(learning_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(batch_loss, global_step=global_step)

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
config = tf.ConfigProto(device_count = {'CPU': 36})

batch_size = 128
parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                         'tensorflow/mnist/logs/mnist_with_summaries'),
    help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()

with tf.Session(config=config) as sess:    
    tf.summary.scalar('precision', precision)
    tf.summary.scalar('recall', recall)
    tf.summary.scalar('fscore', fscore)
    tf.summary.scalar('loss', batch_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    train_dev_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train_dev', sess.graph)
    dev_writer = tf.summary.FileWriter(FLAGS.log_dir + '/dev', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)

    sess.run(init)
    sess.run(init_l)
    sess.run(tf.assign(embedding_weights, EMBEDDING))
    losses = []
    data_size = train_left.shape[0]
    batch_num = data_size / batch_size

    for e in range(10):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_left = train_left[shuffle_indices]
        shuffled_middle = train_middle[shuffle_indices]
        shuffled_right = train_right[shuffle_indices]
        shuffled_mx = train_dep_mx[shuffle_indices]
        shuffled_path = train_path_mx[shuffle_indices]
        shuffled_labels = train_labels[shuffle_indices]
        
        for i in range(batch_num):
            batch_left = shuffled_left[i*batch_size:(i+1)*batch_size]
            batch_middle = shuffled_middle[i*batch_size:(i+1)*batch_size]
            batch_right = shuffled_right[i*batch_size:(i+1)*batch_size]
            batch_mx = shuffled_mx[i*batch_size:(i+1)*batch_size]
            batch_path = shuffled_path[i*batch_size:(i+1)*batch_size]
            batch_labels = shuffled_labels[i*batch_size:(i+1)*batch_size]
            
            if i % 100 == 0:
                print('\nepoch {}, batch {}, loss {}'.format(e, i, np.mean(losses)))

                for name, eval_data, eval_label in [
                    ('train',
                     (batch_left, batch_middle, batch_right, batch_mx, batch_path),
                     batch_labels,
                    ),
                    ('train_dev',
                     (tdev_left, tdev_middle, tdev_right, tdev_dep_mx, tdev_path_mx),
                     tdev_labels),
                    ('dev',
                     (dev_left, dev_middle, dev_right, dev_dep_mx, dev_path_mx),
                     dev_labels),
                    ('test',
                     (test_left, test_middle, test_right, test_dep_mx, test_path_mx),
                     test_labels)
                ]:
                    eval_left, eval_middle, eval_right, eval_mx, eval_path = eval_data
                    p, r, f, l, s = sess.run(
                        [precision, recall, fscore, batch_loss, merged_summary],
                        feed_dict={
                            left_placeholder: eval_left,
                            middle_placeholder: eval_middle,
                            right_placeholder: eval_right,
                            dep_placeholder: eval_mx,
                            path_placeholder: eval_path,
                            label_placeholder: eval_label,
                            keep_prob: 0,
                            keep_prob_dense: 0,
                            training_placeholder: False
                        })
                    if name == 'train':
                        train_writer.add_summary(s, e*batch_num+i)
                    elif name == 'train_dev':
                        train_dev_writer.add_summary(s, e*batch_num+i)
                    elif name == 'dev':
                        dev_writer.add_summary(s, e*batch_num+i)
                    elif name == 'test':
                        test_writer.add_summary(s, e*batch_num+i)

                    print('{}: prec {}, recall {}, fscore {}, loss {}'.format(
                        name, p, r, f, l))

                losses = []

            _, mini_loss = sess.run([train_op, batch_loss], feed_dict={
                left_placeholder: batch_left,
                middle_placeholder: batch_middle,
                right_placeholder: batch_right,
                dep_placeholder: batch_mx,
                path_placeholder: batch_path,
                label_placeholder: batch_labels,
                keep_prob: 0.3,
                keep_prob_dense: 0.2,
                training_placeholder: True
            })

            losses.append(mini_loss)
                
        if DEBUG:
            break
