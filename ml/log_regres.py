"""
logistic regression
"""
import random
from matplotlib import pyplot as plt
from numpy import exp, mat, ones, array, arange


def plot_best_fit(weights):
    """
    plot best fit
    """
    data_mat, cls_mat = load_dataset()
    data_arr = array(data_mat)
    row_size = data_arr.shape[0]

    x_cord1 = []; y_cord1 = []
    x_cord2 = []; y_cord2 = []

    for i in xrange(row_size):
        if int(cls_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')

    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


def load_dataset():
    """
    load data set
    """
    data_mat = []; label_mat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))

    return data_mat, label_mat


def sigmoid(in_x):
    """
    sigmoid
    """
    return 1.0 / (1 + exp(-in_x))


def grad_ascent(data, cls_labels):
    """
    gradient ascent
    """
    data_mat = mat(data)
    # transpose row to column
    label_mat = mat(cls_labels).transpose()
    m, n = data_mat.shape
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in xrange(max_cycles):
        # sum all feauters value up and caculate sigmoid
        # sigmoid value should be [0, 1], it could be used
        # for offset of 0 or 1 class
        h = sigmoid(data_mat * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_mat.transpose() * error

    return weights


def stoc_grade_ascent(data_mat, cls_labels, num_iter=150):
    """
    update incremental for weights when new data comes in
    """
    row_size, col_size = data_mat.shape
    weights = ones(col_size)
    for j in xrange(num_iter):
        data_index = range(row_size)
        for i in xrange(row_size):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            error = cls_labels[rand_index] - h
            weights = weights + alpha * error * data_mat[rand_index]
            del(data_index[rand_index])

    return weights


def classify_vector(vector, weights):
    """
    classify vector
    """
    prob = sigmoid(sum(vector * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    """
    test horse colic with logistic regression classification
    """
    fr_train = open('data/horseColicTraining.txt')
    fr_test = open('data/horseColicTest.txt')
    training_set = []; training_clses = []
    for line in fr_train.readlines():
        cur_line = line.strip().split('\t')
        line_arr = []
        for i in xrange(21):
            line_arr.append(float(cur_line[i]))
        training_set.append(line_arr)
        training_clses.append(float(cur_line[21]))

    training_weights = stoc_grade_ascent(array(training_set), training_clses, 500)
    error_count = 0; num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        cur_line  = line.strip().split('\t')
        line_arr = []
        for i in xrange(21):
            line_arr.append(float(cur_line[i]))
        if int(classify_vector(array(line_arr), training_weights)) != int(cur_line[21]):
            error_count += 1
    error_rate = (float(error_count) / num_test_vec)
    print "the error rate of this test is: %f" % error_rate

    return error_rate


def multi_test():
    """
    mulit test
    """
    num_tests = 10; error_sum = 0.0
    for k in xrange(num_tests):
        error_sum += colic_test()

    print "after %d iterations the avrage error rate is %f" % (num_tests, error_sum/float(num_tests))
