from __future__ import print_function
from data_utils_new import load_sentence_matrix, pad_and_prune_seq
import tensorflow as tf
import numpy as np


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
    print(sent_data[0])
    sent_mx, max_sent_len, sent_padding = sent_data
    head_mx, max_head_len, head_padding = head_data
    dep_mx, max_dep_len, dep_padding = dep_data

    writer = tf.python_io.TFRecordWriter(target)
    for sent, head, dep, label in zip(sent_mx, head_mx, dep_mx, labels):
        sent, sent_len = pad_and_prune_seq(sent, max_sent_len, sent_padding)
        head, head_len = pad_and_prune_seq(head, max_head_len, head_padding)
        dep, dep_len = pad_and_prune_seq(dep, max_dep_len, dep_padding)

        all_seq = np.concatenate((sent, head, dep), axis=0)
        print(np.shape(all_seq))        
        
        all_len = [sent_len, head_len, dep_len]
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq': _float_feature(np.ravel(all_seq)),
            'seq_len': _float_feature(all_len),
            'label': _float_feature(label),
        }))

        writer.write(example.SerializeToString())
    writer.close()


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

if __name__ == '__main__':
    #build_dataset('data/ppi_train.txt', 'data/ppi_train.tfrecords')
    #build_dataset('data/ppi_train_dev.txt', 'data/ppi_train_dev.tfrecords')
    #build_dataset('data/ppi_dev.txt', 'data/ppi_dev.tfrecords')
    build_dataset('data/aimed_training.txt', 'data/aimed_training_p.tfrecords')
    #build_dataset('data/aimed_dev.txt', 'data/aimed_dev.tfrecords')
    #iter_dataset('data/ppi_train.tfrecords', 5, 128)
