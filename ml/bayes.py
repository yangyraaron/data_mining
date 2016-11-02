"""
bayes classification based on probability disitribution
"""
import re
import random
from numpy import zeros, ones, log10, array


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


def bag_of_words_to_vector(vocab_list, input_set):
    """
    convert words to vector
    """
    vector = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            vector[vocab_list.index(word)] += 1
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


def text_parse(str_value):
    """
    text parse
    """
    tokens = re.split(r'\W*', str_value)
    return [tok for tok in tokens if len(tok) > 2]


def spam_test():
    """
    test spam
    """
    docs = []; cls_list = []; full_text=[]
    for i in xrange(1, 26):
        file_path = 'data/spam/%d.txt' % i
        word_list = text_parse(open(file_path).read())
        docs.append(word_list)
        full_text.extend(word_list)
        cls_list.append(1)
        file_path = 'data/ham/%d.txt' % i
        word_list = text_parse(open(file_path).read())
        docs.append(word_list)
        full_text.extend(word_list)
        cls_list.append(0)

    vocab_list = create_vocab_list(docs)
    training_set = range(50); test_set = []
    for i in xrange(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_mat = []; train_cls = []
    for doc_index in training_set:
        train_mat.append(bag_of_words_to_vector(vocab_list, docs[doc_index]))
        train_cls.append(cls_list[doc_index])

    p0v, p1v, pspam = train(array(train_mat), array(train_cls))
    error_count = 0
    for doc_index in test_set:
        word_vector = bag_of_words_to_vector(vocab_list, docs[doc_index])
        if classify(array(word_vector), p0v, p1v, pspam) != cls_list[doc_index]:
            error_count += 1

    print 'error rate is : %f' % (float(error_count) / len(test_set))
