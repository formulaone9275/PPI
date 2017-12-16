from __future__ import print_function
import re
import sys
import codecs
from collections import OrderedDict, Counter
from tqdm import tqdm
sys.path.append('../nlputils')

import spacy
import numpy as np
from biotmreaders import simple_sgml
from range_helper import RangeHelper
from constant import *
from embedding_utils import VOCAB


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

    
# Tokenization.
NLP = spacy.load('en')
NLP.pipeline = [NLP.tagger]
sgml_reader = simple_sgml.SGMLParser()


def map_to_index(tokens):
    indexes = []
    for t in tokens:
        try:
            indexes.append(VOCAB[t])
        except KeyError:
            indexes.append(0)
        #if indexes[-1] == 0:
        #    print(t)
    return indexes


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
                    ele = 'ROOT'

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


def get_dep_relation_rep(relation):
    index = DEP_RELATION_VOCAB.get(relation, 0)
    vec = [0] * DEP_RELATION_VOCAB_SIZE
    vec[index] = 1
    return vec
        
        
def read_sentences(filename):
    sentences, dep_tokens, dep_relations, labels = [], [], [], []
    count = 0
    
    with codecs.open(filename, encoding='utf8') as f:
        for line in tqdm(f):
            count += 1
            if DEBUG and count > 2000:
                break

            line = line.strip()
            info, tagged = line.split('\t')

            # Get tokens.
            doc = sgml_reader.parse(tagged)
            s_doc = NLP(doc['text'])
            t1, t2 = sorted(doc['entity'].values(), key=lambda t:t['charStart'])

            pos = 0
            token_entity = [[0, 1]] * len(s_doc)
            for index, token in enumerate(s_doc):
                curr = doc['text'].find(token.text, pos)

                if RangeHelper.overlap((t1['charStart'], t1['charEnd']),
                                       (curr, curr+len(token.text)-1)):
                    if 'tokenStart' not in t1:
                        t1['tokenStart'] = index
                    t1['tokenEnd'] = index
                    token_entity[index] = [1, 0]

                if RangeHelper.overlap((t2['charStart'], t2['charEnd']),
                                       (curr, curr+len(token.text)-1)):
                    if 'tokenStart' not in t2:
                        t2['tokenStart'] = index
                    t2['tokenEnd'] = index
                    token_entity[index] = [1, 0]
                    
                pos = curr + len(token.text)

            # Rare case: sgml parsing error.
            if 'tokenStart' not in t1 or 'tokenStart' not in t2:
                continue
                
            tokens = []
            for index, token in enumerate(s_doc):
                # Token entity distance.
                to_t1_t2 = [0, 0]
                for t_index, t in enumerate([t1, t2]):
                    if index <= t['tokenStart']:
                        to_t1_t2[t_index] = index - t['tokenStart']
                    elif index >= t['tokenEnd']:
                        to_t1_t2[t_index] = index - t['tokenEnd']

                tokens.append((token.text, token.tag_,
                               token_entity[index], to_t1_t2))

            features = info.split(' ')

            dep_seq = [f for f in features if f.startswith('dep_seq:')][0]
            seq = dep_seq[8:].strip()
            seq_tokens = []
            if len(seq) > 0:
                seq_tokens = seq.split('|')

            sent_dep_relations = []
            dep_path = [f for f in features if f.startswith('dep_path:')][0]
            dep_path = dep_path.split('|')[1]
            dep_path_ele = DEP_ARROW.split(dep_path)
            for ele_i, ele in enumerate(dep_path_ele):
                if len(ele) == 0:
                    continue
                elif ele == '->' or ele == '<-':
                    continue

                if ele == '__':
                    ele = 'ROOT'
                    direction = [0, 0]
                elif dep_path_ele[ele_i-1] == '<-':
                    direction = [1, 0]
                elif dep_path_ele[ele_i-1] == '->':
                    direction = [0, 1]
                else:
                    print(ele, len(ele), dep_path_ele[ele_i-1], dep_path, dep_path_ele)

                ele_index = DEP_RELATION_VOCAB.get(ele, 0)
                
                #sent_dep_relations.append([ele_index] + direction)
                vec = get_dep_relation_rep(ele)
                sent_dep_relations.append(vec + direction)
                    
            sentences.append(tokens)
            dep_tokens.append(seq_tokens)
            dep_relations.append(sent_dep_relations)
            label = features[0]
            # Get label.
            labels.append([0, 1] if label.startswith('R_') else [1, 0])

    return sentences, dep_tokens, dep_relations, np.array(labels)


def sentence_matrix(sentences, max_length=MAX_LENGTH):
    matrix = []
    token_count = 0
    missing_embed = 0
    for sent in sentences:
        if len(sent) == 0:
            tokens, token_pos, token_entity, token_distant = [], [], [], []
        else:
            tokens, token_pos, token_entity, token_distance = zip(*sent)

        indexes = map_to_index(tokens)
        token_count += len(tokens)
        missing_embed += len([t for t in indexes if t==0])
        indexes += [0] * (max_length - len(indexes))
        indexes = indexes[:max_length]
        
        encoded_pos = [encode_pos(t) for t in token_pos]
        encoded_pos += [[0]*8] * (max_length - len(encoded_pos))
        encoded_pos = encoded_pos[:max_length]

        token_entity = list(token_entity)
        token_entity += [[0, 0]] * (max_length - len(token_entity))
        token_entity = token_entity[:max_length]

        token_distance = list(token_distance)
        token_distance += [[0, 0]] * (max_length - len(token_distance))
        token_distance = token_distance[:max_length]
        encoded_distance = [encode_distance(p1) + encode_distance(p2)
                           for p1, p2 in token_distance]

        sent_matrix = [[c1]+c2+c3+c4 for c1, c2, c3, c4 in
                       zip(indexes, encoded_pos, token_entity, encoded_distance) ]
        matrix.append(sent_matrix)

    print('sentences: {}, tokens: {}, missing embedding: {} {}'.format(
        len(sentences), token_count,
        missing_embed, float(missing_embed)/token_count))
    return np.array(matrix)


def dep_seq_matrix(dep_tokens, max_length=20):
    matrix = []
    for tokens in dep_tokens:
        indexes = map_to_index(tokens)
        indexes += [0] * (max_length - len(indexes))
        indexes = indexes[:max_length]

        token_distance = []
        for i, token in enumerate(tokens):
            token_distance.append([i+1, len(tokens)-i]) 
        token_distance += [[0, 0]] * (max_length - len(token_distance))
        token_distance = token_distance[:max_length]
        encoded_distance = [encode_distance(p1) + encode_distance(p2)
                           for p1, p2 in token_distance]
        dep_matrix = [[c1] + c2 for c1, c2 in zip(indexes, encoded_distance)]
        matrix.append(dep_matrix)
    return np.array(matrix)


def dep_path_matrix(dep_relations, max_length=20):
    matrix = []
    for relations in dep_relations:
        relation_distance = []
        for i, r in enumerate(relations):
            relation_distance.append([i+1, len(relations)-i]) 
        relation_distance += [[0, 0]] * (max_length - len(relation_distance))
        relation_distance = relation_distance[:max_length]
        encoded_distance = [encode_distance(p1) + encode_distance(p2)
                           for p1, p2 in relation_distance]

        relations += [[0] * (DEP_RELATION_VOCAB_SIZE+2)] * (max_length - len(relations))
        relations = relations[:max_length]
        
        dep_matrix = [c1 + c2 for c1, c2 in zip(relations, encoded_distance)]
        matrix.append(dep_matrix)
        
    return np.array(matrix)

        
def context_matrix(sentences):
    left, middle, right = [], [], []
    for sent in tqdm(sentences):
        sent_left, sent_middle, sent_right = [], [], []
        for token, pos, entity, distance in sent:
            d1, d2 = distance
            if -6 < d1 <= 0:
                sent_left.append((token, pos, entity, distance))

            if d1 > 0 and d2 < 0:
                sent_middle.append((token, pos, entity, distance))

            if 6 > d2 >= 0:
                sent_right.append((token, pos, entity, distance))
                
        left.append(sent_left)
        middle.append(sent_middle)
        right.append(sent_right)

    matrix_left = sentence_matrix(left, 20)
    matrix_middle = sentence_matrix(middle, 80)
    matrix_right = sentence_matrix(right, 20)
        
    return matrix_left, matrix_middle, matrix_right


def load_context_matrix(filename):
    sents, dep_tokens, dep_path, labels = read_sentences(filename)
    left, middle, right = context_matrix(sents)
    dep_mx = dep_seq_matrix(dep_tokens)
    path_mx = dep_path_matrix(dep_path)
    return left, middle, right, dep_mx, path_mx, labels


if __name__ == '__main__':
    #count_dep_relation('data/ppi_train.txt')
    sentences, dep_tokens, dep_relations, labels = read_sentences('data/ppi_train.txt')

    matrix = sentence_matrix(sentences)
    dep_matrix = dep_seq_matrix(dep_tokens)
    path_matrix = dep_path_matrix(dep_relations)
    #print(list(path_matrix[0]))

    np.savez('data/ppi_train_sent', matrix=matrix,
             dep_matrix=dep_matrix, path_matrix=path_matrix, labels=labels)
    
    left, middle, right = context_matrix(sentences)
    np.savez('data/ppi_train_cont',
             left=left, middle=middle, right=right,
             dep_matrix=dep_matrix, path_matrix=path_matrix, labels=labels)

