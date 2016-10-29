"""
K-nearest neighbars
"""
import os
from numpy import array, tile, zeros
import operator


def create_data_set():
    """
    create data set
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    return group, labels


def classify(in_x, data_set, labels, k):
    """
    classify
    """
    # # of rows
    size = data_set.shape[0]
    # caculate Euclidian distance between input x and items in data set
    # extend input item to same size of data set
    diff_mat = tile(in_x, (size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    # sum a^2 + b^2 + ....
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # end caculate Euclidian distance
    # get sorted index array
    sorted_indicies = distances.argsort()
    class_count = {}
    # find nearest k items from dataset
    for i in xrange(k):
        vote_label = labels[sorted_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # sort the result from largest to smallest
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter,
                                reverse=True)

    return sorted_class_count[0][0]


def file_to_matrix(file_name):
    """
    parse a file and store in a matrix
    """
    file_path = os.path.join('data', file_name)
    if not os.path.exists(file_path):
        print("the file path {} isn't existed".format(file_path))
        return None, None

    fr = open(file_path)
    length = len(fr.readlines())
    mat = zeros((length, 3))
    class_vector = []
    fr = open(file_path)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        cols = line.split('\t')
        # only store first 3 columns
        mat[index, :] = cols[0:3]
        # the last column is the class label
        class_vector.append(int(cols[-1]))
        index += 1
    
    return mat, class_vector
