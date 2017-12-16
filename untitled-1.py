from __future__ import print_function

import tensorflow as tf
import numpy as np

#input_data_all_t=tf.random_normal((20,160,354),0,2)
  #input_data_all_p=tf.placeholder(tf.float32, [None, 160, 354])
  
input_data_all=np.random.rand(120,160,354)
#input_data_all=tf.expand_dims(input_data_all_t,axis=3)

label_t=[0,1]*60+[1,0]*60

label=tf.reshape(label_t,[120,2])
with tf.Session() as sess:
    print(label[0:60].eval())
print(label[0:60].get_shape())
    
x = tf.placeholder(tf.float32, shape=[None, 160, 354])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
  

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=tf.expand_dims(x,axis=3),
    filters=4,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)
  
# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,354 ], strides=1)

print(pool1.get_shape())
# Dense Layer
pool2_flat = tf.reshape(pool1, [-1, 160* 4])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(
    inputs=dense, rate=0.5)

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=2)
y = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        label_batch=label[0:60]
        #batch_t = {0:input_data_all,1:label}
        train_accuracy = accuracy.eval(feed_dict={
                x: input_data_all[0:60], y_: label_batch.eval()})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: input_data_all[0:60], y_: label_batch.eval()})
  
    #print('test accuracy %g' % accuracy.eval(feed_dict={
     #   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))    