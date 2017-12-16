from __future__ import print_function
import re
import sys
import codecs
from collections import OrderedDict, Counter
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from range_helper import RangeHelper
from constant import *
from embedding_utils import VOCAB
from nxml2txt.src import rewriteu2a
from flags import *


U2A_MAPPING = rewriteu2a.load_mapping()

DEP_ARROW = re.compile(r'(<-|->)')

DISTANCE_ENCODING = []
for i in range(10):
    DISTANCE_ENCODING.append([0] * (9-i) + [1] * i)

DISTANCE_MAPPING = OrderedDict([
    (0, DISTANCE_ENCODING[0]),
    (1, DISTANCE_ENCODING[1]),
    (2, DISTANCE_ENCODING[2]),
    (3, DISTANCE_ENCODING[3]),
    (4, DISTANCE_ENCODING[4]),
    (5, DISTANCE_ENCODING[5]),
    (6, DISTANCE_ENCODING[6]),
    (11, DISTANCE_ENCODING[7]),
    (21, DISTANCE_ENCODING[8]),
    (31, DISTANCE_ENCODING[9]),
])


def encode_distance(distance):
    sign = 1 if distance > 0 else 0
    distance = abs(distance)

    if 0 <= distance <= 5:
        return [sign] + DISTANCE_MAPPING[distance]
    elif 6 <= distance <= 10:
        return [sign] + DISTANCE_MAPPING[6]
    elif 11 <= distance <= 20:
        return [sign] + DISTANCE_MAPPING[11]
    elif 21 <= distance <= 30:
        return [sign] + DISTANCE_MAPPING[21]
    elif 31 <= distance:
        return [sign] + DISTANCE_MAPPING[31]

    
POS_ENCODING = []
for i in range(8):
    encoding = [0] * 8
    encoding[i] = 1
    POS_ENCODING.append(encoding)
    
POS_MAPPING = {
    'NN': POS_ENCODING[0],
    'NNS': POS_ENCODING[0],
    'NNP': POS_ENCODING[0],
    'NNPS': POS_ENCODING[0],
    'NNZ': POS_ENCODING[0],
    'NFP': POS_ENCODING[0],

    'VB': POS_ENCODING[1],
    'VBG': POS_ENCODING[1],
    'VBD': POS_ENCODING[1],
    'VBN': POS_ENCODING[1],
    'VBP': POS_ENCODING[1],
    'VBZ': POS_ENCODING[1],

    'TO': POS_ENCODING[2],
    'IN': POS_ENCODING[2],
    'CC': POS_ENCODING[2],

    'RB': POS_ENCODING[3],
    'RBS': POS_ENCODING[3],
    'RBR': POS_ENCODING[3],
    'JJ': POS_ENCODING[3],
    'JJS': POS_ENCODING[3],
    'JJR': POS_ENCODING[3],
    'DT': POS_ENCODING[3],
    'PDT': POS_ENCODING[3],
    'MD': POS_ENCODING[3],
    'MDN': POS_ENCODING[3],

    'WP': POS_ENCODING[4],
    'WP$': POS_ENCODING[4],
    'WRB': POS_ENCODING[4],
    'WDT': POS_ENCODING[4],
    'EX': POS_ENCODING[4],

    'PRP': POS_ENCODING[5],
    'PRP$': POS_ENCODING[5],

    'POS': POS_ENCODING[6],
    'RP': POS_ENCODING[6],
    'FW': POS_ENCODING[6],
    'HYPH': POS_ENCODING[6],
    'SYM': POS_ENCODING[6],

    'CD': POS_ENCODING[7],
}


def encode_pos(pos_tag):
    if pos_tag in POS_MAPPING:
        return POS_MAPPING[pos_tag]
    else:
        return [0]*8


def map_to_index(token):
    try:
        return VOCAB[token]
    except KeyError:
        return 0


def count_dep_relation(filename):
    dep_relation_count = Counter()
    with codecs.open(filename, encoding='utf8') as f:
        for line in tqdm(f):
            line = line.strip()
            info, tagged = line.split('\t')

            features = info.split(' ')

            dep_path = [f for f in features if f.startswith('dep_path:')][0]
            dep_path = dep_path.split('|')[1]
            dep_path_ele = DEP_ARROW.split(dep_path)
            for ele_i, ele in enumerate(dep_path_ele):
                if len(ele) == 0:
                    continue
                elif ele == '->' or ele == '<-':
                    continue
                elif ele == '__':
                    ele = 'root'

                dep_relation_count[ele] += 1

    for rel, count in dep_relation_count.most_common():
        print(rel.encode('utf8'), count)


DEP_RELATION_VOCAB = {}
with codecs.open('dep_relation_count.txt', 'r', encoding='utf8') as f:
    for line in f:
        relation, count = line.strip().split()
        count = int(count)
        if count < 5:
            break
        DEP_RELATION_VOCAB[relation] = len(DEP_RELATION_VOCAB) + 1
DEP_RELATION_VOCAB_SIZE = len(DEP_RELATION_VOCAB)+1


def get_dep_relation_rep(relations):
    vec = [0] * DEP_RELATION_VOCAB_SIZE
    for relation in relations:
        index = DEP_RELATION_VOCAB.get(relation, 0)
        vec[index] = 1
    return vec


def encode_entity_type(ent_type, other_encoding):
    if ent_type == 'O':
        return [1, 0, 0, 0]
    elif ent_type == 'P':
        if other_encoding == 'entity':
            return [0, 1, 0, 0]
        elif other_encoding == 'normal':
            return [1, 0, 0, 0]
    elif ent_type == 'P1':
        return [0, 0, 1, 0]
    elif ent_type == 'P2':
        return [0, 0, 0, 1]


def read_sentences(filename):
    sentences, head_sents, dep_sents, labels = [], [], [], []
    count = 0
    
    with codecs.open(filename, encoding='utf8') as f:
        for line in tqdm(f):
            count += 1
            if DEBUG and count > 10000:
                break

            line = line.strip()
            info, tagged = line.split('\t')
            eles = info.split(' ')
            label = eles[0]
            sent = []
            head = []
            
            tokens = [e for e in eles if e.startswith('token:')]
            for token in tokens:
                token = token[6:]
                word, pos, ent_type, to_e1, to_e2, dep_labels, dep_heads = token.split('|')
                to_e1 = int(to_e1)
                to_e2 = int(to_e2)
                dep_labels = dep_labels.split('@')
                dep_heads = dep_heads.split('@')
                sent.append((word, pos, ent_type, to_e1, to_e2, dep_labels))
                head += dep_heads
            # Get label.
            head_sent = []
            for hd in head:
                if len(hd.strip()) == 0:
                    continue
                hd = int(hd)
                token = sent[hd]
                head_sent.append(token)

            dep_sent = []
            tokens = [e for e in eles if e.startswith('dep_token:')]
            dep_len = [e for e in eles if e.startswith('dep_path_len:')][0]
            dep_len = int(dep_len[13:])
            for index, token in enumerate(tokens):
                token = token[10:]
                word, pos, ent_type, dep_labels, dep_heads = token.split('|')
                dep_labels = dep_labels.split('@')
                dep_heads = dep_heads.split('@')
                dep_sent.append((word, pos, ent_type,
                                 i, i-dep_len, dep_labels))
            sentences.append(sent)
            head_sents.append(head_sent)
            dep_sents.append(dep_sent)
            labels.append([0, 1] if label.startswith('R_') else [1, 0])

    return sentences, head_sents, dep_sents, np.array(labels)


def sentence_matrix(sentences,
                    mask_p1p2=tf.flags.FLAGS.mask_p1p2,
                    mask_other=tf.flags.FLAGS.mask_other,
                    other_encoding=tf.flags.FLAGS.other_encoding):
    matrix = []
    token_count = 0
    missing_embed = 0
    missing_mapped = 0
    for sent in sentences:
        sent_mx = []
        for word, pos, ent_type, to_e1, to_e2, dep_labels in sent:
            token_count += 1
            if mask_p1p2 and (ent_type == 'P1' or ent_type == 'P2'):
                word = '_PROTEIN_'
            if mask_other and ent_type == 'P':
                word = '_PROTEIN_'
            index = map_to_index(word)

            if index == 0:
                missing_embed += 1
                word = [rewriteu2a.mapchar(c, U2A_MAPPING) for c in word]
                word = ''.join(word)
                index = map_to_index(word)
                if index == 0:
                    missing_mapped += 1
            encoded_pos = encode_pos(pos)
            encoded_ent = encode_entity_type(ent_type, other_encoding)
            encoded_to_e1 = encode_distance(to_e1)
            encoded_to_e2 = encode_distance(to_e2)
            encoded_dep = get_dep_relation_rep(dep_labels)
            sent_mx.append([index] + encoded_ent + encoded_pos +
                           encoded_to_e1+encoded_to_e2+encoded_dep)
            '''
            print(word, pos, ent_type, to_e1, to_e2, dep_labels,
                  index, encoded_pos, encoded_ent, encoded_to_e1, encoded_to_e2, encoded_dep)
            '''
        '''
        if len(sent_mx) < max_length:
            sent_mx += [[0] * (33+DEP_RELATION_VOCAB_SIZE)]*(max_length-len(sent_mx))
        '''
        
        matrix.append(sent_mx)

    print('sentences: {}, tokens: {}, missing embedding: {} {}, missing mapped: {} {}'.format(
        len(sentences), token_count,
        missing_embed, float(missing_embed)/token_count,
        missing_mapped, float(missing_mapped)/token_count))
    return np.array(matrix)


def context_matrix(sentences):
    left, middle, right = [], [], []
    for sent in sentences:
        sent_left, sent_middle, sent_right = [], [], []
        for word, pos, ent_type, to_e1, to_e2, dep_labels in sent:
            if to_e1 <=0:
                sent_left.append((word, pos, ent_type, to_e1, to_e2, dep_labels))
            if to_e1 > 0 and to_e2 < 0:
                sent_middle.append((word, pos, ent_type, to_e1, to_e2, dep_labels))
            if 0 <= to_e2:
                sent_right.append((word, pos, ent_type, to_e1, to_e2, dep_labels))
        # Put entities at the first.
        sent_left.reverse()
        left.append(sent_left)
        middle.append(sent_middle)
        right.append(sent_right)

    matrix_left = sentence_matrix(left)
    matrix_middle = sentence_matrix(middle)
    matrix_right = sentence_matrix(right)
    
    return matrix_left, matrix_middle, matrix_right


def load_sentence_matrix(filename):
    
    sentences, head_sents, dep_sents, labels = read_sentences(filename)
    sent_mx = sentence_matrix(sentences)
    #print(np.shape(sent_mx))
    #print(sent_mx[0])
    head_mx = sentence_matrix(head_sents)
    dep_mx = sentence_matrix(dep_sents)
    return [[sent_mx, 160, [0]*155],
            [head_mx, 160, [0]*155],
            [dep_mx, 20, [0]*155]], labels


def load_context_matrix(filename):
    sentences, head_sents, dep_sents, labels = read_sentences(filename)
    left_mx, middle_mx, right_mx = context_matrix(sentences)
    dep_mx = sentence_matrix(dep_sents)
    return [[left_mx, 20, [0]*155],
            [middle_mx, 80, [0]*155],
            [right_mx, 20, [0]*155],
            [dep_mx, 20, [0]*155]], labels


def load_tagged(filename):
    tagged_sents = []
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            info, tagged = line.split('\t')
            tagged_sents.append(tagged)
            
    return tagged_sents


def pad_and_prune_seq(seq, max_len, padding):
    seq_len = max_len if len(seq) > max_len else len(seq)
    seq = seq + [padding] * (max_len - len(seq))
    seq = seq[:max_len]
    
    return seq, seq_len


def create_tensor(data, max_len, padding):
    new_len = []
    new_instances = []
    for instance in data:
        new_instance, length = pad_and_prune_seq(instance, max_len, padding)
        new_instances.append(new_instance)
        new_len.append(length)
    return np.array(new_instances), new_len

                             
def batch_iter(data, labels, epoch_num, batch_size):
    data_size = data[0][0].shape[0]
    batch_num = data_size / batch_size

    for e in range(epoch_num):
        # Shuffle instances.
        shuffle_indices = np.random.permutation(np.arange(data_size))
        for datum in data:
            datum[0] = datum[0][shuffle_indices]
        labels = labels[shuffle_indices]
        
        for i in range(batch_num):
            batch_data = []
            batch_labels = labels[i * batch_size:(i + 1) * batch_size]
            for datum, max_len, padding in data:
                batch = datum[i * batch_size:(i + 1) * batch_size]
                batch, batch_len = create_tensor(batch, max_len, padding)
                batch_data.append((batch, batch_len))
            batch_step = e * batch_num + i
            yield e, i, batch_step, batch_data, batch_labels


if __name__ == '__main__':
    #sent_mx, head_mx, dep_mx, labels = load_sentence_matrix('data2/ppi_train.txt')
    #np.savez('data2/ppi_train_sent', sent_mx=sent_mx, head_mx=head_mx, dep_mx=dep_mx, labels=labels)
    #np.set_printoptions(threshold='nan')
    left_mx, middle_mx, right_mx, dep_mx, labels = load_context_matrix('data2/ppi_train.txt')
    np.savez('data2/ppi_train_cont2', left_mx=left_mx, middle_mx=middle_mx,
             right_mx=right_mx, dep_mx=dep_mx, labels=labels)
    #print(dep_mx.shape)

    

