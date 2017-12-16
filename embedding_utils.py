from gensim.models import KeyedVectors
import numpy as np
from constant import *


# Load word embeddings.
if DEBUG:
    w2v_model = KeyedVectors.load_word2vec_format(
        'data2/PubMed-w2v.bin', binary=True, limit=10000)
else:
    w2v_model = KeyedVectors.load_word2vec_format(
        'data2/PubMed-w2v.bin', binary=True)
   
VOCAB_SIZE = len(w2v_model.vocab)+1
EMBEDDING_DIM = w2v_model.vector_size
EMBEDDING = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

VOCAB = {}
for i in range(1, VOCAB_SIZE):
    word = w2v_model.index2word[i-1]
    VOCAB[word] = i
    
    vector = w2v_model[word]
    EMBEDDING[i] = vector

del w2v_model
