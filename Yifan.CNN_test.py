from __future__ import print_function
from data_utils_new import load_sentence_matrix, pad_and_prune_seq
import tensorflow as tf
import numpy as np
from embedding_utils import VOCAB_SIZE, EMBEDDING
import sklearn as sk
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _parse_function(example_proto):
    features = {
        'seq': tf.FixedLenFeature([52700,], tf.float32),
        'seq_len': tf.FixedLenFeature([3,], tf.float32),
        'label': tf.FixedLenFeature([2,], tf.float32),
    }
    parsed = tf.parse_single_example(example_proto, features)
    seq = tf.reshape(parsed['seq'], [340, 155])
    sent, head, dep = tf.split(seq, [160, 160, 20], axis=0)
    sent_len, head_len, dep_len = tf.split(parsed['seq_len'], [1, 1, 1])
    return sent, sent_len, head, head_len, dep, dep_len, parsed['label']


def build_dataset(filename, target):
    data, labels = load_sentence_matrix(filename)
    
    sent_data, head_data, dep_data = data
    #print(sent_data[0])
    sent_mx, max_sent_len, sent_padding = sent_data
    head_mx, max_head_len, head_padding = head_data
    dep_mx, max_dep_len, dep_padding = dep_data
    input_data=[]
    label_data=[]
    #writer = tf.python_io.TFRecordWriter(target)
    for sent, head,  label in zip(sent_mx, head_mx, labels):
        sent, sent_len = pad_and_prune_seq(sent, max_sent_len, sent_padding)
        head, head_len = pad_and_prune_seq(head, max_head_len, head_padding)
        #dep, dep_len = pad_and_prune_seq(dep, max_dep_len, dep_padding)

        all_seq = np.concatenate((sent, head), axis=0)
                
        
        all_len = [sent_len, head_len]
        input_data.append(all_seq)
        label_data.append(label)
    return input_data,label_data
    '''
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq': _float_feature(np.ravel(all_seq)),
            'seq_len': _float_feature(all_len),
            'label': _float_feature(label),
        }))

        writer.write(example.SerializeToString())
    writer.close()
'''

def iter_dataset(sess, filename, epoch=None, batch_size=None):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    if epoch is not None:
        dataset = dataset.repeat(epoch)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    while True:
        try:
            batch = sess.run(next_element)
            sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, labels = batch
            sent_len = np.ravel(sent_len)
            head_len = np.ravel(head_len)
            dep_len = np.ravel(dep_len)
            yield sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, labels
        except tf.errors.OutOfRangeError:
            break
def cnn_model_fn(features, labels, mode):
    
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 160, 354, 1])
  
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=400,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
  
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 354], strides=2)
  
  
    # Dense Layer
    pool2_flat = tf.reshape(pool1, [-1, 160 * 400])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
  
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
  
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
  
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
  
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
        loss=loss,
          global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def read_data(filename):
    input_data,label=build_dataset(filename, 'data/aimed_training_p.tfrecords')
    #print(np.shape(label))
    #print(label[0:100])
    #concantenate the embedding vector   
    input_data_all_sen=[]
    input_data_all_head=[]    
    for ii in range(len(input_data)):
        input_data_all_temp_sen=[]
        input_data_all_temp_head=[]
        for jj in range(len(input_data[0])):
            if jj<160:
                temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
                #print(temp)
                input_data_all_temp_sen.append(temp)
            else:
                temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
                #print(temp)
                input_data_all_temp_head.append(temp)                
                
        input_data_all_sen.append(input_data_all_temp_sen)
        input_data_all_head.append(input_data_all_temp_head)

    label_list=[]
    for kk in range(len(label)):
        label_list.append(list(label[kk]))
    #label_t=tf.reshape(label_list,[len(label),2])  
    #print(input_data_all.get_shape())    
    return input_data_all_sen,input_data_all_head,label_list

def read_data_v2(filename):
    input_data,label=build_dataset(filename, 'data/aimed_training_p.tfrecords')
    #print(np.shape(label))
    #print(label[0:100])
    #concantenate the embedding vector   
    input_data_all=[]
        
    for ii in range(len(input_data)):
        input_data_all_temp=[]
        
        for jj in range(len(input_data[0])):
            temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
            input_data_all_temp.append(temp)                
                
        input_data_all.append(input_data_all_temp)
       

    label_list=[]
    for kk in range(len(label)):
        label_list.append(list(label[kk]))
    #label_t=tf.reshape(label_list,[len(label),2])  
    #print(input_data_all.get_shape())    
    return input_data_all,label_list
if __name__ == '__main__':
  
    input_data_all, label_list= read_data_v2('data/aimed_training.txt')
    input_data_all_test,label_list_test= read_data_v2('data/aimed_test.txt')   
    
    print(label_list_test)
    x = tf.placeholder(tf.float32, shape=[None, 320, 354])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
      
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=tf.expand_dims(x,axis=3),
        filters=400,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
      
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,354], strides=1)
    
    print(pool1.get_shape())
    # Dense Layer
    pool2_flat = tf.reshape(pool1, [-1, 320* 400])
    #
    print(pool2_flat.get_shape())
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.layers.dropout(
        inputs=dense, rate=keep_prob)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    y = tf.nn.softmax(logits)
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(7e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #define trus positive, false positive, true negative, false negative
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_p = tf.argmax(y, 1)
    y_t = tf.argmax(y_, 1)
    #calculate the precision, recall and F score
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y_, 1), predictions=y_p)
    rec, rec_op = tf.metrics.recall(labels=tf.argmax(y_, 1), predictions=y_p)
    pre, pre_op = tf.metrics.precision(labels=tf.argmax(y_, 1), predictions=y_p)    

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(4):
            
            #batch_t = {0:input_data_all,1:label}
            ce = cross_entropy.eval(feed_dict={
                    x: input_data_all[i*100:i*100+10], y_: label_list[i*100:i*100+10], keep_prob: 1.0})
            print('step %d, cross_entropy %g' % (i, ce),)
            
            
            train_step.run(feed_dict={x: input_data_all[i*100:i*100+10], y_: label_list[i*100:i*100+10], keep_prob: 0.5})
            
            
        #one way to calculate precision recall F score
        y_pred=y_p.eval(feed_dict={x: input_data_all_test, y_: label_list_test, keep_prob: 1.0})
        y_true=y_t.eval(feed_dict={x: input_data_all_test, y_: label_list_test, keep_prob: 1.0})
        print("y_pred:")        
        for nn in range(len(list(y_pred))):
            print(y_pred[nn],",")
        print("y_true:")
        for nn in range(len(list(y_true))):
            print(y_true[nn],",")        
        print("Accuracy", sk.metrics.accuracy_score(y_true, y_pred))
        print("Precision", sk.metrics.precision_score(y_true, y_pred))
        print("Recall", sk.metrics.recall_score(y_true, y_pred))
        print("f1_score", sk.metrics.f1_score(y_true, y_pred))        
        print('test accuracy %g' % accuracy.eval(feed_dict={x: input_data_all_test, y_: label_list_test, keep_prob: 1.0}))     
        
        #Second way to calculate precision reall F score
        v = sess.run(acc_op, feed_dict={x: input_data_all_test, y_: label_list_test, keep_prob: 1.0}) #accuracy
        r = sess.run(rec_op, feed_dict={x: input_data_all_test, y_: label_list_test, keep_prob: 1.0}) #recall
        p = sess.run(pre_op, feed_dict={x: input_data_all_test, y_: label_list_test, keep_prob: 1.0}) #precision
        
        print("accuracy ", v)
        print("recall ", r)
        print("precision ", p)        