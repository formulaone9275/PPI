from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from constant import *
from data_utils import create_input
from embedding_utils import EMBEDDING


train_data, raw_train = create_input('ppi_train.txt')
# 5000 from train.
train_dev_data, raw_train_dev = create_input('ppi_train_dev.txt')
dev_data, raw_dev = create_input('ppi_dev.txt')
test_data, raw_test = create_input('aimed_dev.txt')

X_dev, X_entity_dev, X_distant_dev, X_pos_dev, Y_dev = zip(*dev_data)
X_train_dev, X_entity_train_dev, X_distant_train_dev, X_pos_train_dev, Y_train_dev = zip(*train_dev_data)
X_test, X_entity_test, X_distant_test, X_pos_test, Y_test = zip(*test_data)

# DNN.
embedding_weights = tf.Variable(
    EMBEDDING, trainable=False, name="embedding_weights",
    dtype=tf.float32)

input_placeholder = tf.placeholder(tf.int32, [None, 160])
embedding_layer = tf.nn.embedding_lookup(embedding_weights,
                                         input_placeholder)

entity_placeholder = tf.placeholder(tf.float32, [None, 160, 2])

distant_placeholder = tf.placeholder(tf.float32, [None, 160, 20])
pos_placeholder = tf.placeholder(tf.float32, [None, 160, 8])

combined_input = tf.concat([embedding_layer, entity_placeholder,
                            distant_placeholder, pos_placeholder],
                           axis=2)

combined_input = tf.expand_dims(combined_input, -1)

regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)

conv1 = tf.layers.conv2d(inputs=combined_input,
                         filters=400,
                         kernel_size=[3, 230],
                         activation=tf.nn.relu,
                         kernel_regularizer=regularizer,
                         padding='valid')
#print(conv1.shape)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[158,1],
                                strides=[1, 1], padding='valid')

pool1_flat = tf.reshape(pool1, [-1, 400])

keep_prob = tf.placeholder(tf.float32)
dropout = tf.layers.dropout(pool1_flat, keep_prob)

dense1 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu,
                         kernel_regularizer=regularizer)
dropout1 = tf.layers.dropout(dense1, keep_prob)

dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu,
                         kernel_regularizer=regularizer)
dropout2 = tf.layers.dropout(dense2, keep_prob)

logits = tf.layers.dense(inputs=dropout2, units=2,
                         kernel_regularizer=regularizer)

label_placeholder = tf.placeholder(tf.int32, [None, 2])
#print(logits.shape)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_placeholder)
batch_loss = tf.reduce_mean(loss)
softmax = tf.nn.softmax(logits)

pred = tf.argmax(tf.nn.softmax(logits), 1)
gold = tf.argmax(label_placeholder, 1)
tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(gold, tf.bool))
fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(gold, tf.bool)))
fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(gold, tf.bool))
precision = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
recall = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))

fscore = precision * recall * 2 / (precision+recall)

optimizer = tf.train.AdamOptimizer(0.0006)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(batch_loss, global_step=global_step)

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
config = tf.ConfigProto(device_count = {'CPU': 20})

with tf.Session(config=config) as sess:    
    sess.run(init)
    sess.run(init_l)
    losses = []
    for e in range(10):
        shuffle_indices = np.random.permutation(np.arange(train_data.shape[0]))
        shuffled_data = train_data[shuffle_indices]

        for i in range(train_data.shape[0]/64):
            batch = shuffled_data[i*64:(i+1)*64]
            X_train, X_entity, X_distant_train, X_pos_train, Y_train = zip(*batch)

            _, mini_loss = sess.run([train_op, batch_loss], feed_dict={
                input_placeholder: X_train,
                entity_placeholder: X_entity,
                distant_placeholder: X_distant_train,
                pos_placeholder: X_pos_train,
                label_placeholder: Y_train,
                keep_prob: 0.7,
            })

            losses.append(mini_loss)

            #if DEBUG:
            #    break
            
            if i % 100 == 0:
                print('\nepoch {}, batch {}, loss {}'.format(e, i, np.mean(losses)))

                # Train devel
                train_dev_p, train_dev_r, train_dev_f, train_dev_l = sess.run(
                    [precision, recall, fscore, batch_loss],
                    feed_dict={
                        input_placeholder: X_train_dev,
                        entity_placeholder: X_entity_train_dev,
                        distant_placeholder: X_distant_train_dev,
                        pos_placeholder: X_pos_train_dev,
                        label_placeholder: Y_train_dev,
                        keep_prob: 1,
                    })
                print('train_dev: prec {}, recall {}, fscore {}, loss {}'.format(
                    train_dev_p, train_dev_r, train_dev_f, train_dev_l))

                # Devel
                dev_p, dev_r, dev_f, dev_l = sess.run(
                    [precision, recall, fscore, batch_loss],
                    feed_dict={
                        input_placeholder: X_dev,
                        entity_placeholder: X_entity_dev,
                        distant_placeholder: X_distant_dev,
                        pos_placeholder: X_pos_dev,
                        label_placeholder: Y_dev,
                        keep_prob: 1,
                    })
                print('dev: prec {}, recall {}, fscore {}, loss {}'.format(
                    dev_p, dev_r, dev_f, dev_l))

                # Test
                test_p, test_r, test_f, test_l = sess.run(
                    [precision, recall, fscore, batch_loss],
                    feed_dict={
                        input_placeholder: X_test,
                        entity_placeholder: X_entity_test,
                        distant_placeholder: X_distant_test,
                        pos_placeholder: X_pos_test,
                        label_placeholder: Y_test,
                        keep_prob: 1,
                    })
                print('test: prec {}, recall {}, fscore {}, loss {}'.format(
                    test_p, test_r, test_f, test_l))
                losses = []

        if DEBUG:
            break
