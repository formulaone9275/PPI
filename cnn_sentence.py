from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from constant import *
from data_utils import read_sentences, sentence_matrix, dep_seq_matrix
from embedding_utils import EMBEDDING

if DEBUG:
    train_sents, train_dep_tokens, train_labels = read_sentences('ppi_train.txt')
    train_data = sentence_matrix(train_sents)
    train_dep_mx = dep_seq_matrix(train_dep_tokens)
else:
    train_set = np.load('ppi_train_sent.npz')
    train_data, train_dep_mx, train_labels = \
        train_set['matrix'], train_set['dep_matrix'], train_set['labels']

tdev_sents, tdev_dep_tokens, tdev_labels = read_sentences('ppi_train_dev.txt')
tdev_data = sentence_matrix(tdev_sents)
tdev_dep_mx = dep_seq_matrix(tdev_dep_tokens)

dev_sents, dev_dep_tokens, dev_labels = read_sentences('ppi_dev.txt')
dev_data = sentence_matrix(dev_sents)
dev_dep_mx = dep_seq_matrix(dev_dep_tokens)

test_sents, test_dep_tokens, test_labels = read_sentences('aimed_dev.txt')
test_data = sentence_matrix(test_sents)
test_dep_mx = dep_seq_matrix(test_dep_tokens)

# DNN.
regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
print(regularizer)
keep_prob = tf.placeholder(tf.float32)
keep_prob_dense = tf.placeholder(tf.float32)

data_placeholder = tf.placeholder(tf.float32, [None, 160, 31])
dep_placeholder = tf.placeholder(tf.float32, [None, 20, 21])

embedding_weights = tf.Variable(
    EMBEDDING, trainable=False, name="embedding_weights",
    dtype=tf.float32)

token, feature = tf.split(data_placeholder, [1, 30], 2)
token = tf.cast(tf.reshape(token, [-1, 160]), tf.int32)
embed = tf.nn.embedding_lookup(embedding_weights, token)

final = tf.concat([embed, feature], axis=2)
final = tf.expand_dims(final, -1)

conv = tf.layers.conv2d(inputs=final,
                        filters=100,
                        kernel_size=[3, 230],
                        activation=tf.nn.relu,
                        kernel_regularizer=regularizer,
                        padding='valid')

pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[158,1],
                               strides=[1, 1], padding='valid')

flat = tf.reshape(pool, [-1, 100])

dep_token, dep_feature = tf.split(dep_placeholder, [1, 20], 2)
dep_token = tf.cast(tf.reshape(dep_token, [-1, 20]), tf.int32)
dep_embed = tf.nn.embedding_lookup(embedding_weights, dep_token)
dep_final = tf.concat([dep_embed, dep_feature], axis=2)
dep_final = tf.expand_dims(dep_final, -1)
dep_conv = tf.layers.conv2d(inputs=dep_final,
                            filters=100,
                            kernel_size=[2, 220],
                            activation=tf.nn.relu,
                            kernel_regularizer=regularizer,
                            padding='valid')

dep_pool = tf.layers.max_pooling2d(inputs=dep_conv, pool_size=[19,1],
                                   strides=[1, 1], padding='valid')
dep_flat = tf.reshape(dep_pool, [-1, 100])

combined = tf.concat([flat, dep_flat], axis=1)
dropout = tf.layers.dropout(combined, keep_prob)
print(combined.shape)

dense1 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu,
                         kernel_regularizer=regularizer)
dropout1 = tf.layers.dropout(dense1, keep_prob_dense)

dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu,
                         kernel_regularizer=regularizer)
dropout2 = tf.layers.dropout(dense2, keep_prob_dense)

logits = tf.layers.dense(inputs=dropout2, units=2,
                         kernel_regularizer=regularizer)

label_placeholder = tf.placeholder(tf.int32, [None, 2])

loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_placeholder)
batch_loss = tf.reduce_mean(loss)

pred = tf.argmax(tf.nn.softmax(logits), 1)
gold = tf.argmax(label_placeholder, 1)
tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(gold, tf.bool))
fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(gold, tf.bool)))
fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(gold, tf.bool))
precision = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
recall = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))

fscore = precision * recall * 2 / (precision+recall)

global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(
    0.0006,                # Base learning rate.
    global_step,  # Current index into the dataset.
    200,          # Decay step.
    0.9,                # Decay rate.
    staircase=True
)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(batch_loss, global_step=global_step)

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
config = tf.ConfigProto(device_count = {'CPU': 20})

batch_size = 64

with tf.Session(config=config) as sess:    
    sess.run(init)
    sess.run(init_l)
    losses = []
    for e in range(10):
        data_size = train_data.shape[0]
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = train_data[shuffle_indices]
        shuffled_mx = train_dep_mx[shuffle_indices]
        shuffled_labels = train_labels[shuffle_indices]
        
        for i in range(data_size/batch_size):
            batch = shuffled_data[i*batch_size:(i+1)*batch_size]
            labels = shuffled_labels[i*batch_size:(i+1)*batch_size]
            mx = shuffled_mx[i*batch_size:(i+1)*batch_size]
            _, mini_loss = sess.run([train_op, batch_loss], feed_dict={
                data_placeholder: batch,
                dep_placeholder: mx,
                label_placeholder: labels,
                keep_prob: 0.6,
                keep_prob_dense: 0.8,
            })

            losses.append(mini_loss)

            if DEBUG:
                break
            
            if i % 100 == 0:
                print('\nepoch {}, batch {}, loss {}'.format(e, i, np.mean(losses)))
                lr, gs = sess.run([learning_rate, global_step])
                print(lr, gs)
                
                for name, eval_data, eval_mx, eval_label in [
                        ('train_dev', tdev_data, tdev_dep_mx, tdev_labels),
                        ('dev', dev_data, dev_dep_mx, dev_labels),
                        ('test', test_data, test_dep_mx, test_labels)
                ]:
                    # Train devel
                    p, r, f, l = sess.run(
                        [precision, recall, fscore, batch_loss],
                        feed_dict={
                            data_placeholder: eval_data,
                            dep_placeholder: eval_mx,
                            label_placeholder: eval_label,
                            keep_prob: 1,
                            keep_prob_dense: 1,
                        })
                    print('{}: prec {}, recall {}, fscore {}, loss {}'.format(
                        name, p, r, f, l))

                losses = []

        if DEBUG:
            break
