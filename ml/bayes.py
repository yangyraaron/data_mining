"""
bayes classification based on probability disitribution
"""
from numpy import zeros, ones, log10


def load_dataset():
    """
    load sample dataset
    """
    posting_list=[['my', 'dog', 'has', 'flea', \
    'problems', 'help', 'please'],
    ['maybe', 'not', 'take', 'him', \
    'to', 'dog', 'park', 'stupid'],
    ['my', 'dalmation', 'is', 'so', 'cute', \
    'I', 'love', 'him'],
    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
    'to', 'stop', 'him'],
    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    class_vector = [0,1,0,1,0,1] #1 is abusive, 0 not

    return posting_list, class_vector


def create_vocab_list(data_set):
    """
    create vocabulary list
    """
    vocab_set = set([])
    for doc in data_set:
        vocab_set = vocab_set | set(doc)

    return list(vocab_set)


def words_to_vector(vocab_list, input_set):
    """
    convert words to vector
    """
    vector = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            vector[vocab_list.index(word)] = 1
        else:
            print "word: %s is not in my vocabulary!" % word
    return vector


def train(matrix, category):
    """
    train
    """
    doc_size = len(matrix)
    words_size = len(matrix[0])

    p_abusive = sum(category) / float(doc_size)
    p0_size = ones(words_size)
    p1_size = ones(words_size)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in xrange(doc_size):
        if category[i] == 1:
            p1_size += matrix[i]
            p1_denom += sum(matrix[i])
        else:
            p0_size += matrix[i]
            p0_denom += sum(matrix[i])

    p1_vector = log10(p1_size / p1_denom)
    p0_vector = log10(p0_size / p0_denom)

    return p0_vector, p1_vector, p_abusive


def classify(vector, p0_vec, p1_vec, p_class1):
    """
    classify
    """
    p1 = sum(vector * p1_vec) + log10(p_class1)
    p0 = sum(vector * p0_vec) + log10(1.0 -  p_class1)
    if p1 > p0:
        return 1
    else:
        return 0
